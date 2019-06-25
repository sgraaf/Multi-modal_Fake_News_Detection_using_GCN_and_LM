#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import pickle as pkl
from pathlib import Path

import numpy as np
import torch

from models import GraphConvolutionalNetwork
from utils import (load_model, load_data, get_user_embeddings)

# defaults
DEVICE = torch.device('cpu')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT_DIR = Path.cwd().parent
DATA_DIR = ROOT_DIR / 'data'
CHECKPOINTS_DIR = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR = ROOT_DIR / 'output' / 'models'
RESULTS_DIR = ROOT_DIR / 'output' / 'results'

HIDDEN_DIM = 16
DROPOUT = 0.5

# def generate():
# load the data
print('Loading the data...')
adj_file = DATA_DIR / 'adj_matrix.npz'
features_file = DATA_DIR / 'features_matrix.pkl'
labels_file = DATA_DIR / 'labels_matrix.pkl'
splits_file = DATA_DIR / 'splits_dict.pkl'
adj, features, labels, splits_dict = load_data(adj_file, features_file, labels_file, splits_file)
train_idxs = splits_dict['train']
val_idxs = splits_dict['val']
test_idxs = splits_dict['test']
all_idxs = torch.cat([train_idxs, val_idxs, test_idxs])

# initialize the model
print('Initializing the model...')
model = GraphConvolutionalNetwork(
        input_dim=features.shape[1], 
        hidden_dim=HIDDEN_DIM, 
        num_classes=labels.max().item() + 1,  
        dropout=DROPOUT
).to(DEVICE)

# load the model
model_file = MODELS_DIR / f'{model.__class__.__name__}_model.pt'
load_model(model_file, model)

# generate GCN embeddings
model.eval()
output = model.get_embeddings(features, adj)

# get article embeddings
all_article_embeds = output[all_idxs].tolist()
train_article_embeds = output[train_idxs].tolist()
val_article_embeds = output[val_idxs].tolist()
test_article_embeds = output[test_idxs].tolist()

all_article_embeds_file = RESULTS_DIR / 'GCN_all_article_embeds_pre_ReLU.pkl'
train_article_embeds_file = RESULTS_DIR / 'GCN_train_article_embeds_pre_ReLU.pkl'
val_article_embeds_file = RESULTS_DIR / 'GCN_val_article_embeds_pre_ReLU.pkl'
test_article_embeds_file = RESULTS_DIR / 'GCN_test_article_embeds_pre_ReLU.pkl'

pkl.dump(all_article_embeds, open(all_article_embeds_file, 'wb'))
pkl.dump(train_article_embeds, open(train_article_embeds_file, 'wb'))
pkl.dump(val_article_embeds, open(val_article_embeds_file, 'wb'))
pkl.dump(test_article_embeds, open(test_article_embeds_file, 'wb'))

# get user embeddings
all_user_embeds = get_user_embeddings(output, adj, all_idxs)
train_user_embeds = get_user_embeddings(output, adj, train_idxs)
val_user_embeds = get_user_embeddings(output, adj, val_idxs)
test_user_embeds = get_user_embeddings(output, adj, test_idxs)

all_user_embeds_file = RESULTS_DIR / 'GCN_all_user_embeds_pre_ReLU.pkl'
train_user_embeds_file = RESULTS_DIR / 'GCN_train_user_embeds_pre_ReLU.pkl'
val_user_embeds_file = RESULTS_DIR / 'GCN_val_user_embeds_pre_ReLU.pkl'
test_user_embeds_file = RESULTS_DIR / 'GCN_test_user_embeds_pre_ReLU.pkl'

pkl.dump(all_user_embeds, open(all_user_embeds_file, 'wb'))
pkl.dump(train_user_embeds, open(train_user_embeds_file, 'wb'))
pkl.dump(val_user_embeds, open(val_user_embeds_file, 'wb'))
pkl.dump(test_user_embeds, open(test_user_embeds_file, 'wb'))


if __name__ == '__main__':
    generate()
