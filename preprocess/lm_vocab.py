import argparse
import nltk
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()

tokenizer = nltk.tokenize.NLTKWordTokenizer()

counts = {}

with open(args.input) as f:
    for line in tqdm(f):
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            token = token.lower()
            if token not in counts:
                counts[token] = 0
            counts[token] += 1


    result_counts = sorted([(key, counts[key]) for key in counts], key = lambda x: -x[1])

    torch.save(result_counts, args.output)
