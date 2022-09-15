import argparse
#import nltk
from conllu import parse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ud_tag', type=str)
parser.add_argument('--out_tag', type=str)

args = parser.parse_args()

vocab_size = 2 ** 14
chunk_size = 10000

counts = {}

ud_root = '/raid/lingo/abau/lm/ud-treebanks-v2.7/%s' % args.ud_tag
files = os.listdir(ud_root)
input_train = os.path.join(ud_root, [f for f in files if f[-12:] == 'train.conllu'][0])
input_dev = os.path.join(ud_root, [f for f in files if f[-10:] == 'dev.conllu'][0])

out_vocab = '/raid/lingo/abau/lm/%s-vocab.pt' % args.out_tag
out_dir = '/raid/lingo/abau/lm/%s-corpus' % args.out_tag

with open(input_train) as f:
    corpus = parse(f.read())

    for tokens in corpus:
        #tokens = tokenizer.tokenize(line)
        for token in tokens:
            token = token['form'].lower()
            if token not in counts:
                counts[token] = 0
            counts[token] += 1

    result_counts = sorted([(key, counts[key]) for key in counts], key = lambda x: -x[1])

    torch.save(result_counts, out_vocab)

# 0 is unknown
# 1 is end-of-sentence
vocab = result_counts[:vocab_size - 2]
vocab_dict = { v: i + 2 for i, (v, c) in enumerate(vocab) }

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Dev chunks first
chunk_number = 0

with open(input_dev) as f:
    corpus = parse(f.read())

    sentences = []

    for line in corpus:
        sentences.append([
            vocab_dict[token['form'].lower()] if token['form'].lower() in vocab_dict else 0
            for token in line
        ] + [1])

        if len(sentences) >= chunk_size:
            print('Storing %d sentences (dev)' % len(sentences))
            # Bucket sentences by length
            sentences_by_length = {}
            for sentence in sentences:
                l = len(sentence)
                if l not in sentences_by_length:
                    sentences_by_length[l] = []
                sentences_by_length[l].append(sentence)

            torch.save(sentences_by_length, os.path.join(out_dir, 'chunk-%d.pt' % (chunk_number,)))

            sentences = []
            chunk_number += 1

    sentences_by_length = {}
    for sentence in sentences:
        l = len(sentence)
        if l not in sentences_by_length:
            sentences_by_length[l] = []
        sentences_by_length[l].append(sentence)

    print('Storing %d sentences (dev)' % len(sentences))
    torch.save(sentences_by_length, os.path.join(out_dir, 'chunk-%d.pt' % (chunk_number,)))
    sentences = []
    chunk_number += 1

# Non-dev chunks next
with open(input_train) as f:
    corpus = parse(f.read())

    sentences = []

    for line in corpus:
        sentences.append([
            vocab_dict[token['form'].lower()] if token['form'].lower() in vocab_dict else 0
            for token in line
        ] + [1])

        if len(sentences) >= chunk_size:
            print('Storing %d sentences (train)' % len(sentences))
            # Bucket sentences by length
            sentences_by_length = {}
            for sentence in sentences:
                l = len(sentence)
                if l not in sentences_by_length:
                    sentences_by_length[l] = []
                sentences_by_length[l].append(sentence)

            torch.save(sentences_by_length, os.path.join(out_dir, 'chunk-%d.pt' % (chunk_number,)))

            sentences = []
            chunk_number += 1

    sentences_by_length = {}
    for sentence in sentences:
        l = len(sentence)
        if l not in sentences_by_length:
            sentences_by_length[l] = []
        sentences_by_length[l].append(sentence)

    print('Storing %d sentences (train)' % len(sentences))
    torch.save(sentences_by_length, os.path.join(out_dir, 'chunk-%d.pt' % (chunk_number,)))
