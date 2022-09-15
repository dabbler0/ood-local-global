import argparse
import torch
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm, trange

MAX_BATCH_SIZE = 128

def outedge(model, sequences, length_dist, beam_size):
    n, seq_len = sequences.shape

    # Omit "bad token"
    outputs, hiddens = pred(model, sequences[:, :-1])

    # Final marginal output has size batch_size x output_size
    final_marginal_output = F.softmax(outputs, dim=1) * length_dist[0]

    # Beam search up to length_dist
    # Each tensor has size batch_size x beam_size x data dimensions...
    hsize = (hiddens.shape[0], hiddens.shape[2])
    hidden_frontier = hiddens.transpose(0, 1).unsqueeze(1).contiguous()
    output_frontier = outputs.unsqueeze(1)
    score_frontier = torch.zeros((n, 1))

    for i in range(1, len(length_dist)):
        # To advanace the beam, first take the top beam_size outputs for
        # each beam
        values, indices = torch.topk(output_frontier, beam_size, dim=2)

        # values, indices should now each be batch_size x beam_size x beam_size

        # Match each new char to its corresponding beam.
        index_batches = indices.reshape(n, -1) # turn into batch_size x beam_size * beam_size
        score_batches = score_frontier.repeat_interleave(beam_size, dim=1) # same
        score_additions = values.reshape(n, -1) # same

        hidden_batches = hidden_frontier.repeat_interleave(beam_size, dim=1) # now batch_size x beam_size * beam_size x data

        new_scores = score_batches + score_additions # batch_size x beam_size * beam_size

        best_scores, best_indices = torch.topk(new_scores, beam_size, dim=1) # batch_size x beam_size
        feed_indices = index_batches[
                torch.arange(index_batches.shape[0]).unsqueeze(1),
                best_indices
        ] # Index to get batch_size x beam_size
        feed_hiddens = hidden_batches[
                torch.arange(hidden_frontier.shape[0]).unsqueeze(1),
                best_indices
        ] # Index to get batch_size x beam_size x data...

        # Advance to get new hiddens and outputs
        new_outputs, new_hiddens = advance(model,
            feed_hiddens.reshape(-1, hsize[0], hsize[1]).transpose(0, 1).contiguous(),
            feed_indices.reshape(-1)
        )

        hidden_frontier = new_hiddens.transpose(0, 1).reshape(n, beam_size, hsize[0], hsize[1]).contiguous()
        output_frontier = new_outputs.reshape(n, beam_size, -1)
        score_frontier = best_scores

        softmaxed_scores = F.softmax(best_scores, dim=1)

        marginal = (F.softmax(output_frontier, dim=2) * softmaxed_scores.unsqueeze(2)).sum(dim=1)

        final_marginal_output += marginal * length_dist[i]

    return final_marginal_output

def advance(model, hiddens, chars):
    n = chars.shape[0]

    new_hiddens = []
    outputs = []

    for i in range(n // MAX_BATCH_SIZE + 1):
        output, hidden = model(
            chars[i * MAX_BATCH_SIZE:(i + 1) * MAX_BATCH_SIZE].cuda(),
            hiddens[:, i * MAX_BATCH_SIZE:(i + 1) * MAX_BATCH_SIZE, :].cuda(),
        )
        new_hiddens.append(hidden.cpu())
        outputs.append(output.cpu())

    return torch.cat(outputs, dim=0), torch.cat(new_hiddens, dim=1)

def pred(model, seq):
    n, seq_len = seq.shape

    hiddens = []
    outputs = []

    for i in range(n // MAX_BATCH_SIZE + 1):
        output, hidden = pred_batch(
            model,
            seq[i * MAX_BATCH_SIZE:(i + 1) * MAX_BATCH_SIZE].cuda()
        )
        hiddens.append(hidden.cpu())
        outputs.append(output.cpu())

    return torch.cat(outputs, dim=0), torch.cat(hiddens, dim=1)

def pred_batch(model, seq):
    batch_size, seq_len = seq.shape

    hidden = model.init_hidden(batch_size).cuda()

    for i in range(seq_len):
        output, hidden = model(seq[:, i], hidden)

    return output, hidden

def extend_by(seq, values):
    return torch.cat([
        seq.unsqueeze(0).repeat(values.shape[0], 1).cuda(),
        values.unsqueeze(1).cuda()
    ], dim=1)

def transformer_outedge(transformer, seq, n = 15):
    seq = seq[:-1]
    initial = torch.nn.functional.softmax(transformer_pred(transformer, seq.unsqueeze(0)))

    # Take top n
    values, indices = torch.topk(initial, n)

    after_result = (torch.nn.functional.softmax(
        transformer_pred(transformer, extend_by(seq, indices)),
        dim=1
    ) * values.unsqueeze(1)).sum(dim=0)

    after_result /= after_result.sum()

    return after_result

def transformer_pred(transformer, seq):
    seq = seq.transpose(0, 1).cuda()
    mask = transformer.generate_square_subsequent_mask(seq.size(0)).cuda()
    return transformer(seq, mask)[-1].squeeze()
