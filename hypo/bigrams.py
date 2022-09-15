import argparse
import torch
import numpy as np
import os
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--data', type=str)
parser.add_argument('--vocab_size', type=int, default=6000)

# Output
parser.add_argument('--output', type=str)

args = parser.parse_args()

# Get inputs
chunks = [os.path.join(args.data, f) for f in os.listdir(args.data) if f != 'chunk-0.pt']
validation_file = os.path.join(args.data, 'chunk-0.pt')

# Get vocab for generation vis

counts = torch.zeros((args.vocab_size, args.vocab_size), dtype=torch.long)
vcounts = torch.zeros((args.vocab_size, args.vocab_size), dtype=torch.long)

for chunk in tqdm(chunks):
    print('Loading', chunk)
    data = torch.load(chunk)

    for length in data:
        pool = data[length]
        for sentence in pool:
            for i, j in zip(sentence[:-1], sentence[1:]):
                counts[i, j] += 1

vdata = torch.load(validation_file)

for length in vdata:
    pool = vdata[length]
    for sentence in pool:
        for i, j in zip(sentence[:-1], sentence[1:]):
            vcounts[i, j] += 1

# Prepare output
torch.save(counts, args.output)

# Bigram entropy
smoothed_bigrams = counts + 1e-6
smoothed_vbigrams = vcounts + 1e-6
unigram_probs = smoothed_bigrams.sum(0)
unigram_probs /= unigram_probs.sum()
voccur_probs = smoothed_vbigrams / smoothed_vbigrams.sum()
conditional_bigrams = smoothed_bigrams / smoothed_bigrams.sum(1, keepdim=True)
print('Held-out bigram entropy', (voccur_probs * torch.log(conditional_bigrams)).sum())
print('Held-out unigram entropy', (voccur_probs * torch.log(unigram_probs.unsqueeze(0))).sum())
