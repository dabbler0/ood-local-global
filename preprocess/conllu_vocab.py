import argparse
#import nltk
from conllu import parse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()

#tokenizer = nltk.tokenize.NLTKWordTokenizer()

counts = {}

with open(args.input) as f:
    corpus = parse(f.read())

    for tokens in corpus:
        #tokens = tokenizer.tokenize(line)
        for token in tokens:
            token = token['form'].lower()
            if token not in counts:
                counts[token] = 0
            counts[token] += 1


    result_counts = sorted([(key, counts[key]) for key in counts], key = lambda x: -x[1])

    torch.save(result_counts, args.output)
