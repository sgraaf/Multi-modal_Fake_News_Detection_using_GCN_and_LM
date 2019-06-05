#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import DocAttentionRNN, SentAttentionRNN


class HierarchicalAttentionNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0):
        super(HierarchicalAttentionNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.sent_attend = SentAttentionRNN(self.input_dim, self.hidden_dim)
        self.doc_attend = DocAttentionRNN(self.hidden_dim * 2, self.hidden_dim)
        self.softmax_classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, batch, batch_dims):
        # print(f'batch shape: {batch.shape}')
        # print(f'batch dims: {batch_dims}')

        # apply dropout to the batch
        batch_dropout = F.dropout(batch, p=self.dropout, training=self.training)

        # get the sentence embeddings
        sent_embeds = self.sent_attend(batch_dropout, batch_dims)

        # get the document embeddings
        doc_embeds = self.doc_attend(sent_embeds, batch_dims)

        # get the classification
        out = self.softmax_classifier(doc_embeds)

        return out
