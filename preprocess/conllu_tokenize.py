import argparse
#import nltk
from conllu import parse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--chunk_size', type=int)
parser.add_argument('--output', type=str)

args = parser.parse_args()

#tokenizer = nltk.tokenize.NLTKWordTokenizer()

# 0 is unknown
# 1 is end-of-sentence
vocab = torch.load(args.vocab_file)[:args.vocab_size - 2]
vocab_dict = { v: i + 2 for i, (v, c) in enumerate(vocab) }

if not os.path.exists(args.output):
    os.mkdir(args.output)
chunk_number = 0

with open(args.input) as f:
    corpus = parse(f.read())

    sentences = []

    for line in corpus:
        sentences.append([
            vocab_dict[token['form'].lower()] if token['form'].lower() in vocab_dict else 0
            for token in line
        ] + [1])

        if len(sentences) >= args.chunk_size:
            print('Storing %d sentences' % len(sentences))
            # Bucket sentences by length
            sentences_by_length = {}
            for sentence in sentences:
                l = len(sentence)
                if l not in sentences_by_length:
                    sentences_by_length[l] = []
                sentences_by_length[l].append(sentence)

            torch.save(sentences_by_length, os.path.join(args.output, 'chunk-%d.pt' % (chunk_number,)))

            sentences = []
            chunk_number += 1

    sentences_by_length = {}
    for sentence in sentences:
        l = len(sentence)
        if l not in sentences_by_length:
            sentences_by_length[l] = []
        sentences_by_length[l].append(sentence)

    torch.save(sentences_by_length, os.path.join(args.output, 'chunk-%d.pt' % (chunk_number,)))
