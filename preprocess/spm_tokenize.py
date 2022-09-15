import argparse
import nltk
import torch
import os
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--sp_model', type=str)
parser.add_argument('--chunk_size', type=int)
parser.add_argument('--output', type=str)

args = parser.parse_args()

tokenizer = nltk.tokenize.NLTKWordTokenizer()
sp = spm.SentencePieceProcessor(model_file=args.sp_model)

if not os.path.exists(args.output):
    os.mkdir(args.output)
chunk_number = 0

with open(args.input) as f:
    sentences = []

    for line in f:
        sentences.append(sp.encode(line))

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

    if len(sentences) > 0:
        print('Storing %d sentences' % len(sentences))
        # Bucket sentences by length
        sentences_by_length = {}
        for sentence in sentences:
            l = len(sentence)
            if l not in sentences_by_length:
                sentences_by_length[l] = []
            sentences_by_length[l].append(sentence)

        torch.save(sentences_by_length, os.path.join(args.output, 'chunk-%d.pt' % (chunk_number,)))
