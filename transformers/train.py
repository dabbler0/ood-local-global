# From the pytorch Transformer tutorials.

import io
import math
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
import numpy as np
from torch import nn
from collections import Counter
from torchtext.vocab import Vocab

from model import TransformerModel

import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embed_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--nheads', type=int, default=4)

args = parser.parse_args()

torch.manual_seed(args.seed)

if not os.path.exists(args.output):
    os.mkdir(args.output)

ntokens = 2 ** 14 #len(vocab.stoi) # the size of vocabulary
emsize = args.embed_size # embedding dimension
nhid = args.hidden_size # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = args.nheads # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

chunks = [os.path.join(args.data, f) for f in os.listdir(args.data) if f != 'chunk-0.pt']
validation_file = os.path.join(args.data, 'chunk-0.pt')

import time

criterion = nn.CrossEntropyLoss()
#lr = 1.0 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    total_batches = 0
    for chunk_num, chunk in enumerate(chunks):
        data = torch.load(chunk)

        batches = []
        for length in data:
            if length <= 1:
                continue
            pool = data[length]
            np.random.shuffle(pool)

            for i in range(len(pool) // args.batch_size + 1):
                if i * args.batch_size < len(pool):
                    strings = torch.LongTensor(
                        pool[i * args.batch_size : (i + 1) * args.batch_size]
                    )

                    batches.append(
                        (strings[:, :-1],
                        strings[:, 1:].reshape(-1))
                    )

        for batch, (data, targets) in enumerate(batches):
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 100
            total_batches += 1
            if total_batches % log_interval == 0 and total_batches > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} chunks | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, chunk_num, len(chunks), 3e-4,#scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    total_length = 0
    with torch.no_grad():
        data = torch.load(validation_file)

        batches = []
        for length in data:
            if length <= 1:
                continue
            pool = data[length]
            np.random.shuffle(pool)

            for i in range(len(pool) // args.batch_size + 1):
                if i * args.batch_size < len(pool):
                    strings = torch.LongTensor(
                        pool[i * args.batch_size : (i + 1) * args.batch_size]
                    )

                    batches.append(
                        (strings[:, :-1],
                        strings[:, 1:].reshape(-1))
                    )

        for data, targets in batches:
            data = data.to(device)
            targets = targets.to(device)

            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += targets.shape[0] * criterion(output_flat, targets).item()
            total_length += targets.shape[0]
    return total_loss / total_length

best_val_loss = float("inf")
epochs = 50 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    torch.save(model.state_dict(), os.path.join(args.output, 'epoch-%d-loss-%f.pt' % (epoch, val_loss)))
    #scheduler.step()
