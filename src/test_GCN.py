#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import GraphConvolutionalNetwork
from utils import (accuracy, load_model, load_data, get_user_embeddings)

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

def test():
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

    # initialize the criterion
    criterion = nn.NLLLoss()

    # generate GCN embeddings
    model.eval()
    output = model(features, adj)
    test_targets = labels[test_idxs].max(dim=1).indices
    test_loss = criterion(output[test_idxs], test_targets)
    test_acc = accuracy(output[test_idxs], test_targets)

    # metrics
    # ...


if __name__ == '__main__':
    test()
