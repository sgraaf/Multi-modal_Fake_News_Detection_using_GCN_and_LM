#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import pickle as pkl
from pathlib import Path

import numpy as np
import torch

from models import GraphConvolutionalNetwork
from utils import (load_model, load_data)

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

def generate():
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
    all_output = output[all_idxs].tolist()
    train_output = output[train_idxs].tolist()
    val_output = output[val_idxs].tolist()
    test_output = output[test_idxs].tolist()
    
    all_file = RESULTS_DIR / 'GCN_all_embeds.pkl'
    train_file = RESULTS_DIR / 'GCN_train_embeds.pkl'
    val_file = RESULTS_DIR / 'GCN_val_embeds.pkl'
    test_file = RESULTS_DIR / 'GCN_test_embeds.pkl'
    
    pkl.dump(all_output, open(all_file, 'wb'))
    pkl.dump(train_output, open(train_file, 'wb'))
    pkl.dump(val_output, open(val_file, 'wb'))
    pkl.dump(test_output, open(test_file, 'wb'))


if __name__ == '__main__':
    generate()
