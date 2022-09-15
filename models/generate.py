import argparse
import torch
import nltk
import numpy as np
import os
from tqdm import tqdm, trange
from model import StandardRNN

eps = 1e-6

def random_softmax(tensor):
    t = torch.rand(1).cuda() - eps
    #tensor[0] = -9999
    weights = torch.nn.functional.softmax(tensor.squeeze(), dim=0)
    sums = torch.cumsum(weights, 0)
    indices = torch.nonzero(sums >= t)
    min_index = torch.min(indices)

    return min_index.cpu().numpy().tolist()

def generate(model, max_len = 50, temperature = 1, init_sequence = [2]):
    model.eval()

    with torch.no_grad():
        string = init_sequence
        hidden = model.init_hidden(1).cuda()

        for c in init_sequence[:-1]:
            _, hidden = model(
                torch.LongTensor([c]).cuda(),
                hidden
            )

        # Generate
        last_char = string[-1]
        for _ in range(max_len):
            output, hidden = model(
                torch.LongTensor([last_char]).cuda(),
                hidden
            )

            last_char = random_softmax(output / temperature)
            string.append(last_char)

            if last_char == 1:
                break

        return string

def beam_generate(model, max_len = 50, beam_size = 5, init_sequence = (2,)):
    model.eval()

    with torch.no_grad():
        string = init_sequence
        hidden = model.init_hidden(1).cuda()

        for c in init_sequence[:-1]:
            _, hidden = model(
                torch.LongTensor([c]).cuda(),
                hidden
            )

        beam_candidates = [
            (string, 0, hidden)
        ]

        for _ in range(max_len):
            new_candidates = []
            for string, score, state in beam_candidates:
                if string[-1] == 1:
                    new_candidates.append(
                        (string, score, state)
                    )

                else:
                    output, hidden = model(
                        torch.LongTensor([string[-1]]).cuda(),
                        state
                    )

                    values, indices = torch.topk(output, beam_size)

                    for v, i in zip(values.squeeze(), indices.squeeze()):
                        new_candidates.append(
                            (string + (i.cpu().numpy().tolist(),),
                            score + v.cpu().numpy().tolist(),
                            hidden)
                        )
            beam_candidates = sorted(new_candidates, key = lambda x: -x[1])[:beam_size]

        return beam_candidates[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--vocab_size', type=int, default=6000)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--init', type=str, default='The')

    args = parser.parse_args()

    model = torch.load(args.model)
    vocab = torch.load(args.vocab)
    vocab_dict = { i + 2: v for i, (v, c) in enumerate(vocab) }
    inv_dict = { v: i + 2 for i, (v, c) in enumerate(vocab) }
    vocab_dict[0] = '<unk>'
    vocab_dict[1] = '<end>'

    tokenizer = nltk.tokenize.NLTKWordTokenizer()

    init_sequence = [
        inv_dict[token.lower()] if token.lower() in inv_dict else 0
        for token in tokenizer.tokenize(args.init)
    ]

    result = generate(model, temperature = args.temp, init_sequence = init_sequence)
    print(' '.join(vocab_dict[x] for x in result))
