import argparse
import torch
import numpy as np
import os
from tqdm import tqdm, trange
from model import StandardRNN
from generate import generate

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--data', type=str)
parser.add_argument('--vocab', type=str)
parser.add_argument('--vocab_size', type=int, default=6000)

# Model options
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--resume', type=str)

# Training options
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--subst_prob', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0)
parser.add_argument('--state_dropout', type=float, default=0)

# Output
parser.add_argument('--output', type=str)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Get inputs
chunks = [os.path.join(args.data, f) for f in os.listdir(args.data) if f != 'chunk-0.pt']
validation_file = os.path.join(args.data, 'chunk-0.pt')

print(chunks)

# Get vocab for generation vis
#vocab = torch.load(args.vocab)[:args.vocab_size - 2]
#vocab_dict = { i + 2: v[0] for i, v in enumerate(vocab) }
#token_dict = { v[0]: i + 2 for i, v in enumerate(vocab) }
#vocab_dict[0] = '<UNK>'
#vocab_dict[1] = '<END>'

#choice_vector = [i + 2 for i, v in enumerate(vocab)]
#p_vector = [v[1] for v in vocab]
#total = sum(p_vector)
#p_vector = [x / total for x in p_vector]

unigram_counts = torch.zeros(args.vocab_size)

print('Counting unigrams for substitutions')
for chunk in tqdm(chunks):
    data = torch.load(chunk)
    for length in data:
        pool = data[length]
        for seq in pool:
            for tok in seq:
                unigram_counts[tok] += 1

unigram_counts /= unigram_counts.sum()

# Substitutions according to unigram counts
def get_random_subst():
    return np.random.choice(list(range(args.vocab_size)), p=unigram_counts)

# Prepare output
if not os.path.exists(args.output):
    os.mkdir(args.output)

# Init model
if args.resume is None:
    model = StandardRNN(args.vocab_size, args.hidden_size, args.vocab_size, n_layers = args.layers,
            embedding_dropout = args.embed_dropout,
            state_dropout = args.state_dropout).cuda()
    num_steps = 0
else:
    checkpoints = os.listdir(args.resume)
    latest = max(checkpoints, key = lambda x: int(x.split('steps-')[1].split('-')[0]))
    num_steps = int(latest.split('steps-')[1].split('-')[0])
    print('Loading', os.path.join(args.resume, latest), num_steps)
    model = torch.load(os.path.join(args.resume, latest))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

for epoch in trange(args.epochs):
    print('--- EPOCH %d --- ' % epoch)

    # --- Training ---
    model.train()

    # Load chunks in a random order
    np.random.shuffle(chunks)
    for chunk_num, chunk in enumerate(tqdm(chunks)):
        model.train()
        data = torch.load(chunk)

        # Create a batch pool
        batches = []

        # Get sentences in batches by length
        for length in data:
            # Shuffle the pool of sentences of each length
            pool = data[length]
            np.random.shuffle(pool)

            # Divide into batches of at most batch_size
            for i in range(len(pool) // args.batch_size + 1):
                if i * args.batch_size < len(pool):
                    batches.append(
                        torch.LongTensor(
                            pool[i * args.batch_size : (i + 1) * args.batch_size]
                        )
                    )

        # Take batches from this chunk in a random order
        np.random.shuffle(batches)

        for batch in tqdm(batches):
            # Randomly replace batch words with other words
            batch_size, seq_len = batch.shape

            if seq_len <= 1:
                continue

            batch = batch.cuda()

            subs = batch.clone()
            if args.subst_prob > 0:
                for i in range(batch_size):
                    for j in range(seq_len):
                        if np.random.random() < args.subst_prob:
                            subs[i, j] = get_random_subst()

            hidden = model.init_hidden(batch_size).cuda()
            loss = 0

            for i in range(seq_len - 1):
                output, hidden = model(subs[:, i], hidden)
                # Scale gradient magnitude by batch size
                loss += criterion(output, batch[:, i + 1]) * batch_size / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Generate for visualization
        model.eval()
        #generated_seq = generate(model, init_sequence=[token_dict['the']])
        #print('Generated: %s' % (' '.join(vocab_dict[x] for x in generated_seq)))

        # --- Validation and saving ---
        if num_steps % 5 == 0 or chunk_num == len(chunks) - 1:
            # Load validation chunk
            validation = torch.load(validation_file)
            valid_loss = 0
            samples = 0

            # Get sentences in batches by length
            for length in validation:
                # Shuffle the pool of sentences of each length
                pool = validation[length]

                # Divide into batches of at most batch_size
                for i in range(len(pool) // args.batch_size + 1):
                    if i * args.batch_size < len(pool):
                        batch = torch.LongTensor(
                            pool[i * args.batch_size : (i + 1) * args.batch_size]
                        ).cuda()

                        # Test on these batches
                        batch_size, seq_len = batch.shape

                        hidden = model.init_hidden(batch_size).cuda()

                        for i in range(seq_len - 1):
                            output, hidden = model(batch[:, i], hidden)
                            valid_loss += criterion(output, batch[:, i + 1]).data * batch_size
                            samples += batch_size

            valid_loss /= samples

            print('Validation loss: %f' % valid_loss)

            torch.save(model, os.path.join(args.output, 'epoch-%d-steps-%d-loss-%f.pt' % (epoch, num_steps, valid_loss)))

        num_steps += 1
