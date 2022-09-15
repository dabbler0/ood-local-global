import torch
from hypo import preds
import numpy as np
import nltk
import matplotlib
from matplotlib import pyplot as plt
import os
import sentencepiece as spm
import sys
from tqdm.notebook import tqdm, trange

def load_best_model_from(root):
    checkpoints = os.listdir(root)
    best = min(checkpoints, key = lambda x: float(x.split('loss-')[1].split('.pt')[0]))
    loss = float(best.split('loss-')[1].split('.')[0])
    return torch.load(os.path.join(root, best))

SEL_CUTOFF = 200
PROB_THRESH = 1. / (SEL_CUTOFF - 2)

class TestingContext:
    def __init__(self, vocab_file, corpus_dir, bigrams_file, vocab_size = 2 ** 14, vocab_type='word', rand_seed = 0):
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        
        chunk0 = os.path.join(corpus_dir, 'chunk-0.pt')

        self.vocab_size = vocab_size
        self.validation_set = torch.load(chunk0)

        self.vocab_type = vocab_type

        if self.vocab_type == 'word':
            self.vocab = torch.load(vocab_file)
            self.tokenizer = nltk.NLTKWordTokenizer()
            self.vocab_dict = { v[0]: i + 2 for i, v in enumerate(self.vocab) if i < vocab_size - 2 }
        else:
            self.sp = spm.SentencePieceProcessor(model_file=vocab_file)

        self.bigrams = torch.load(bigrams_file)
        self.bigram_probs = (self.bigrams + 1e-6) / (self.bigrams + 1e-6).sum(1, keepdim=True)
        self.unigram = self.bigrams.sum(1) / self.bigrams.sum()
        self.smooth_bigrams = self.bigrams + self.unigram.unsqueeze(0)
        self.smooth_bigram_probs = self.smooth_bigrams / self.smooth_bigrams.sum(1, keepdim=True)

        self.trunc_pool = []
        for length in self.validation_set:
            for seq in self.validation_set[length]:
                trunc_len = np.random.randint(1, length)
                self.trunc_pool.append(seq[:trunc_len])

        self.trunc_subpool = np.random.choice(self.trunc_pool, 200, replace=False)
        self.subpool = None

        self.unigram_preds = torch.stack(
            [ self.unigram for seq in self.trunc_subpool ]
        ).squeeze()

    def generate_surprises(self, model):
        self.subpool = [self.generate_surprise(model, x) for x in self.trunc_subpool]
        self.bigram_preds = torch.stack(
            [ self.bigram_next(seq) for seq in self.subpool ]
        ).squeeze()
        self.smooth_bigram_preds = torch.stack(
            [ self.smooth_bigram_next(seq) for seq in self.subpool ]
        ).squeeze()
        return self.subpool

    def generate_surprise(self, model, x):
        predictions = preds.pred(model, torch.LongTensor(x).unsqueeze(0))[0].squeeze()
        # This gives prediction vector
        # Take something from the top 200 that with probability
        # at most 1/198.
        indices = (predictions > PROB_THRESH)[2:SEL_CUTOFF].nonzero()
        return x + [int(indices[torch.randint(len(indices), (1,))])]

    def bigram_next(self, seq):
        return self.bigram_probs[seq[-1]]

    def smooth_bigram_next(self, seq):
        return self.smooth_bigram_probs[seq[-1]]

    def int2tok(self, c):
        return self.vocab[c - 2][0] if c >= 2 else '<unk>' if c == 0 else '<end>'

    def seq2sen(self, seq):
        return ' '.join(self.int2tok(c) for c in seq)

    def sen2seq(self, sen):
        if self.vocab_type == 'word':
            return torch.LongTensor([
                self.vocab_dict[token.lower()] if token.lower() in self.vocab_dict else 0
                for token in self.tokenizer.tokenize(sen)
            ])
        else:
            return torch.LongTensor(self.sp.encode(sen))

    def run_test_transformer(self, eval_model):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        total_length = 0
        max_batch_size = 128
        with torch.no_grad():
            data = self.validation_set

            batches = []
            for length in data:
                if length <= 1:
                    continue
                pool = data[length]
                np.random.shuffle(pool)

                for i in range(len(pool) // max_batch_size + 1):
                    if i * max_batch_size < len(pool):
                        strings = torch.LongTensor(
                            pool[i * max_batch_size : (i + 1) * max_batch_size]
                        )

                        batches.append(
                            (strings[:, :-1],
                            strings[:, 1:].reshape(-1))
                        )

            for data, targets in batches:
                data = data.cuda()
                targets = targets.cuda()

                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = eval_model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += targets.shape[0] * criterion(output_flat, targets).item()
                total_length += targets.shape[0]
        return total_loss / total_length

    def run_test(self, model):
        max_batch_size = 128
        valid_loss = 0
        samples = 0
        criterion = torch.nn.CrossEntropyLoss()

        for length in tqdm(self.validation_set):
            # Shuffle the pool of sentences of each length
            pool = self.validation_set[length]

            # Divide into batches of at most batch_size
            for i in range(len(pool) // max_batch_size + 1):
                if i * max_batch_size < len(pool):
                    batch = torch.LongTensor(
                        pool[i * max_batch_size : (i + 1) * max_batch_size]
                    ).cuda()

                    # Test on these batches
                    batch_size, seq_len = batch.shape

                    hidden = model.init_hidden(batch_size).cuda()

                    for i in range(seq_len - 1):
                        output, hidden = model(batch[:, i], hidden)
                        valid_loss += criterion(output, batch[:, i + 1]).data * batch_size
                        samples += batch_size

        valid_loss /= samples
        return valid_loss

    def get_bigram_loss(self):
        true_bigrams = torch.zeros((self.vocab_size, self.vocab_size))
        valid_loss = 0
        samples = 0

        for length in tqdm(self.validation_set):
            # Shuffle the pool of sentences of each length
            pool = self.validation_set[length]

            for sentence in pool:
                for char, targ in zip(sentence[:-1], sentence[1:]):
                    true_bigrams[char, targ] += 1

        eps = 1e-6
        smoothed_bigrams = (self.bigrams + eps).float()
        smoothed_bigram_log_probs = torch.log(smoothed_bigrams) - torch.log(smoothed_bigrams.sum(dim=1, keepdim=True))

        return (true_bigrams * smoothed_bigram_log_probs).sum() / true_bigrams.sum()

    def outedge_for_model(self, model):
        return torch.stack(
            [ preds.outedge(model, torch.LongTensor(seq).unsqueeze(0), [0, 1], 15) for seq in tqdm(self.subpool) ]
        ).squeeze()

    def actual_for_model(self, model):
        return torch.nn.functional.softmax(torch.stack(
            [ preds.pred(model, torch.LongTensor(seq).unsqueeze(0))[0] for seq in tqdm(self.subpool) ]
        ).squeeze(), dim=1)

    def ignore_for_model(self, model):
        return torch.nn.functional.softmax(torch.stack(
            [ preds.pred(model, torch.LongTensor(seq[:-1]).unsqueeze(0))[0] for seq in tqdm(self.subpool) ]
        ).squeeze(), dim=1)

    def outedge_for_transformer(self, transformer):
        return torch.stack(
            [ preds.transformer_outedge(transformer, torch.LongTensor(seq), 15) for seq in tqdm(self.subpool) ]
        ).squeeze()

    def actual_for_transformer(self, transformer):
        return torch.nn.functional.softmax(torch.stack(
            [ preds.transformer_pred(transformer, torch.LongTensor(seq).unsqueeze(0)) for seq in tqdm(self.subpool) ]
        ).squeeze(), dim=1)

    def ignore_for_transformer(self, transformer):
        return torch.nn.functional.softmax(torch.stack(
            [ preds.transformer_pred(transformer, torch.LongTensor(seq[:-1]).unsqueeze(0)) for seq in tqdm(self.subpool) ]
        ).squeeze(), dim=1)

    def add_interpolation_loss(self, outedge, actual, outedge_weight, method = 'tvd'):
        return self.dist(outedge * outedge_weight + self.bigram_preds * (1 - outedge_weight), actual, method = method)

    def mul_interpolation_loss(self, outedge, actual, outedge_weight, method = 'tvd'):
        return self.dist(
            torch.nn.functional.softmax(
                torch.log(outedge) * outedge_weight +
                torch.log(self.bigram_preds + 1e-6) * (1 - outedge_weight),
                dim = 1
            ),
            actual,
            method = method
        )

    def l1d(self, a, b, true, t):
        magnitudes = a - b
        signs = torch.sign(t * a + (1 - t) * b - true)
        return -torch.sum(magnitudes * signs)

    def optimal_add_interpolation(self, outedge_preds, actual_preds, method = 'tvd'):
        outedge_preds = outedge_preds.cpu()
        actual_preds = actual_preds.cpu()
        t = 0.5
        n = 1
        for _ in range(10):
            n += 1
            dv = self.l1d(outedge_preds, self.bigram_preds, actual_preds, t)
            if dv > 0:
                t += 1 / (2 ** n)
            else:
                t -= 1 / (2 ** n)
        return t, self.add_interpolation_loss(outedge_preds, actual_preds, t, method = method)

    def optimal_mul_interpolation(self, outedge_preds, actual_preds, method = 'tvd'):
        outedge_preds = outedge_preds.cpu()
        actual_preds = actual_preds.cpu()
        t = min(np.linspace(0, 1, 50), key = lambda x: self.mul_interpolation_loss(outedge_preds, actual_preds, x))
        return t, self.mul_interpolation_loss(outedge_preds, actual_preds, t, method = method)

    def add_interpolation_curve(self, outedge_preds, actual_preds, ticks=50):
        x, y = np.linspace(0, 1, ticks), [self.mul_interpolation_loss(outedge_preds, actual_preds, a) for a in np.linspace(0, 1, ticks)]
        plt.plot(x, y)
        return x, y

    def mul_interpolation_curve(self, outedge_preds, actual_preds, ticks=50):
        x, y = np.linspace(0, 1, ticks), [self.mul_interpolation_loss(outedge_preds, actual_preds, a) for a in np.linspace(0, 1, ticks)]
        plt.plot(x, y)
        return x, y

    def dist(self, a, b, method = 'tvd'):
        a = a.cpu()
        b = b.cpu()

        if method == 'tvd':
            return torch.abs(a - b).sum(dim = 1).mean() / 2
        elif method == 'js':
            # Add eps for logarithms
            eps = 1e-6
            m = (a + b) / 2

            loga = torch.log(a + eps)
            logb = torch.log(b + eps)
            logm = torch.log(m + eps)

            return (a * (loga - logm) + b * (logb - logm)).sum(dim = 1).mean() / 2

    def all_hyp_for_transformer(self, transformer, copycat_model = None, gold_model = None):
        if gold_model is None:
            gold_model = copycat_model

        outedge = self.outedge_for_transformer(gold_model).cpu()
        actual = self.actual_for_transformer(transformer).cpu()
        ignore = self.ignore_for_transformer(transformer).cpu()

        result = {
            'unigram': self.dist(self.unigram_preds, actual),
            'bigram': self.dist(self.bigram_preds, actual),
            'outedge': self.dist(outedge, actual),
            'ignore': self.dist(ignore, actual),
            'add_interp': self.optimal_add_interpolation(outedge, actual)[1],
            'mul_interp': self.optimal_mul_interpolation(outedge, actual)[1]
        }

        if copycat_model is not None:
            copycat = self.actual_for_transformer(copycat_model)
            result['copycat'] = self.dist(copycat, actual)

        return result

    def all_hyp_for_model(self, model, copycat_model = None, gold_model = None, method = 'tvd'):
        if gold_model is None:
            gold_model = copycat_model

        outedge = self.outedge_for_model(gold_model).cpu()
        actual = self.actual_for_model(model).cpu()
        ignore = self.ignore_for_model(model).cpu()

        result = {
            'unigram': self.dist(self.unigram_preds, actual, method),
            'bigram': self.dist(self.bigram_preds, actual, method),
            'outedge': self.dist(outedge, actual, method),
            'ignore': self.dist(ignore, actual, method),
            'add_interp': self.optimal_add_interpolation(outedge, actual, method)[1],
            'mul_interp': self.optimal_mul_interpolation(outedge, actual, method)[1]
        }

        if copycat_model is not None:
            copycat = self.actual_for_model(copycat_model).cpu()
            result['copycat'] = self.dist(copycat, actual, method)

        return result

    def visualize_prediction(self, v):
        values, indices = torch.topk(v, 5)
        return [
            (v, self.int2tok(i))
            for v, i in zip(values, indices)
        ]

    def display_hypotheses_for(self, model, sen):
        seq = self.sen2seq(sen).unsqueeze(0)

        outedge = preds.outedge(model, seq, [0, 1], 15)
        actual = preds.pred(model, seq)[0]
        bigram = self.bigram_next(seq[0])

        return {
            'outedge': self.visualize_prediction(outedge),
            'actual': self.visualize_prediction(actual),
            'bigram': self.visualize_prediction(bigram)
        }
