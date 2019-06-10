#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle as pkl

import torch
import torch.nn.functional as F
import torch.utils.data as data

from allennlp.modules.elmo import batch_to_ids

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class FNNDataset(data.Dataset):

    def __init__(self, file_path, GloVe_vectors, ELMo):
        super(FNNDataset, self).__init__()
        self.GloVe = GloVe_vectors
        self.ELMo = ELMo
        self.articles, self.labels = self.load_data(file_path)

    def __getitem__(self, idx):
        article = self.articles[idx]
        # print(type(article))
        label = self.labels[idx]

        sent_lens = [len(sentence) for sentence in article]
        MAX_SENT_LEN = max(sent_lens)

        # get the GloVe embeddings
        if self.GloVe is not None:
            GloVe_embed = torch.stack([torch.stack([self.GloVe[word] if word in self.GloVe.stoi else self.GloVe[word.lower()] for word in sent + ['<pad>'] * (MAX_SENT_LEN - len(sent))]) for sent in article])
        # print(GloVe_embeddings.shape)

        # get the ELMo embeddings
        if self.ELMo is not None:
            ELMo_character_ids = batch_to_ids(article).to(DEVICE)
            ELMo_embed = self.ELMo(ELMo_character_ids)['elmo_representations'][0]
        # print(ELMo_embeddings.shape)

        # concat the GloVe and ELMo embeddings
        if self.ELMo is not None and self.GloVe is not None:
            article_embed = torch.cat([GloVe_embed, ELMo_embed], dim=2)
        elif self.ELMo is not None:
            article_embed = ELMo_embed
        elif self.GloVe is not None:
            article_embed = GloVe_embed

        return article_embed, sent_lens, label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_data(file_path):
        dataset = pkl.load(open(file_path, 'rb'))
        return dataset['articles'], dataset['labels']

    @property
    def size(self):
        return len(self.labels)


class PadSortBatchFNN(object):
    def __call__(self, batch):
                # sort the batch
        batch_sorted_sent = sorted(
            batch, key=lambda x: x[0].shape[1], reverse=True)
        batch_sorted_doc = sorted(
            batch_sorted_sent, key=lambda x: x[0].shape[0], reverse=True)

        # unpack the batch
        articles_sorted, article_dims_sorted, labels_sorted = map(list, zip(*batch_sorted_doc))

        # pad the articles with zeroes
        max_doc_len = max([article.shape[0] for article in articles_sorted])
        max_sent_len = max([article.shape[1] for article in articles_sorted])
        articles_padded = [F.pad(article, (0, 0, 0, max_sent_len - article.shape[1], 0, max_doc_len - article.shape[0])) for article in articles_sorted]

        # convert articles and labels to tensors
        articles_tensor = torch.stack(articles_padded)
        labels_tensor = torch.LongTensor(labels_sorted)

        return articles_tensor, article_dims_sorted, labels_tensor
