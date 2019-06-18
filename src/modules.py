#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as Pack
from torch.nn.utils.rnn import pad_packed_sequence as Pad


class SentAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(SentAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # initialize the LSTM cell
        self.LSTM_cell = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # initialize the attention parameters
        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.attend_weights = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, batch, batch_dims):
        doc_lens = [len(dim) for dim in batch_dims]
        max_doc_len = max(doc_lens)
        sents_lens = batch_dims
        sent_embeds = []

        for i in range(batch.shape[0]):
            article_len = doc_lens[i]
            sent_lens = torch.LongTensor(sents_lens[i])

            # remove pad sentences
            article = batch[i][:article_len]

            # sort
            sent_lens_sorted, sort_idxs = torch.sort(sent_lens, dim=0, descending=True)
            article_sorted = article[sort_idxs, :]

            # pack the article (batch)
            article_packed = Pack(article_sorted, sent_lens_sorted, batch_first=True)

            # run the packed article (batch) through the LSTM cell
            article_encoded_packed, _ = self.LSTM_cell(article_packed)

            # unpack the article (batch)
            article_encoded, article_encoded_lens = Pad(article_encoded_packed, batch_first=True)
            if torch.cuda.is_available():
                article_encoded, article_encoded_lens = article_encoded.cuda(), article_encoded_lens.cuda()

            # compute the attention
            hidden_embed = torch.tanh(self.linear(article_encoded))
            attention = self.attend_weights(hidden_embed).squeeze(-1)
            mask = torch.arange(attention.shape[1])[None, :] < article_encoded_lens[:, None]  # create the mask
            attention[~mask] = -float('inf')  # mask the attention
            masked_softmax = torch.softmax(attention, dim=1).unsqueeze(-1)  # perform softmax
            attended = masked_softmax * article_encoded

            # sum to obtain sentence representations
            attended_sum = attended.sum(1, keepdim=True).squeeze(1)

            # pad to stack
            attended_pad = F.pad(attended_sum, (0, 0, 0, max_doc_len - attended_sum.shape[0]))

            sent_embeds.append(attended_pad)

        return torch.stack(sent_embeds)


class DocAttentionRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DocAttentionRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # initialize the LSTM cell
        self.LSTM_cell = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # initialize the attention parameters
        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.attend_weights = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, batch, batch_dims):
        doc_lens = torch.LongTensor([len(dim) for dim in batch_dims])

        # sort
        doc_lens_sorted, sort_idxs = torch.sort(doc_lens, dim=0, descending=True)
        batch_sorted = batch[sort_idxs, :]

        # pack the batch
        batch_packed = Pack(batch_sorted, doc_lens_sorted, batch_first=True)

        # run the packed batch through the LSTM cell
        batch_encoded_packed, _ = self.LSTM_cell(batch_packed)

        # unpack the batch
        batch_encoded, batch_encoded_lens = Pad(batch_encoded_packed, batch_first=True)
        if torch.cuda.is_available():
            batch_encoded, batch_encoded_lens = batch_encoded.cuda(), batch_encoded_lens.cuda()

        # compute the attention
        hidden_embed = torch.tanh(self.linear(batch_encoded))
        attention = self.attend_weights(hidden_embed).squeeze(-1)
        mask = torch.arange(attention.shape[1])[None, :] < batch_encoded_lens[:, None]  # create the mask
        attention[~mask] = -float('inf')  # mask the attention
        masked_softmax = torch.softmax(attention, dim=1).unsqueeze(-1)  # perform softmax
        attended = masked_softmax * batch_encoded

        # sum to obtain doc representations
        attended_sum = attended.sum(1, keepdim=True).squeeze(1)

        return attended_sum


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Implementation based on https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))

        self.reset_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) + self.bias
        
        return output 

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})'
