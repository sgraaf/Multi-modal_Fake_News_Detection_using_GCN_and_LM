#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import argparse
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import GraphConvolutionalNetwork
from utils import (create_directories, print_flags, print_model_parameters,
                   load_data)

# defaults
FLAGS = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

SEED = 42
ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.01
MAX_EPOCHS = 200
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 16
DROPOUT = 0.5

np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_TYPE_DEFAULT = 'glove_and_elmo'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
RUN_DESC_DEFAULT = None
    
def train_epoch_fn(train_iter, model, optimizer, loss_func_fn):
    train_loss = 0.0
    train_acc = []
    for step, batch in enumerate(train_iter):
        articles, article_dims, labels = batch
        if step % 50 == 0 and step != 0:
            print(f'Processed {step} FN batches.')
            print(f'Accuracy: {train_acc[len(train_acc)-1]}.')
        optimizer.zero_grad()
        out = model(batch=articles, batch_dims=article_dims)
        loss = loss_func_fn(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * BATCH_SIZE_FN
        acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()
        train_acc.append(acc)
    return train_loss, train_acc

def eval_epoch_fn(val_iter, model, loss_func_fn):
    val_acc = []
    val_loss = 0.0
    for step, batch in enumerate(val_iter):
        articles, article_dims, labels = batch
        out = model(batch=articles, batch_dims=article_dims)
        loss = loss_func_fn(out, labels)
        val_loss += loss.item() * BATCH_SIZE_FN
        acc = (out.argmax(dim=1).to(DEVICE) == labels.to(DEVICE)).float().mean()
        val_acc.append(acc)
    return val_loss, val_acc

    
def train():
    model_type = FLAGS.model_type
    run_desc = FLAGS.run_desc
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir) / model_type / run_desc
    models_dir = Path(FLAGS.models_dir) / model_type / run_desc
    results_dir = Path(FLAGS.results_dir) / model_type / run_desc
    learning_rate = LEARNING_RATE

    if not data_dir.exists():
        raise ValueError('Data directory does not exist')
    
    # create other directories if they do not exist
    create_directories(checkpoints_dir, models_dir, results_dir)
    
    # load the data
    print('Loading the data...')
    adj_file = data_dir / 'adj_matrix.npz'
    features_file = data_dir / 'features_matrix.pkl'
    labels_file = data_dir / 'labels_matrix.pkl'
    adj, features, labels = load_data(adj_file, features_file, labels_file)

    # initialize the model, according to the model type
    print('Initializing the model...')
    model = GraphConvolutionalNetwork(
            input_dim=features.shape[1], 
            hidden_dim=HIDDEN_DIM, 
            num_classes=labels.max().item() + 1,  
            dropout=DROPOUT
    ).to(DEVICE)
    print_model_parameters(model)

    # set the criterion and optimizer
    print('Initializing the criterion and optimizer)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )        

    # initialize the results dict
    results = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
    }    

    print(f'Starting training at epoch 0...')
    for i in range(epoch, MAX_EPOCHS):
        print(f'Epoch {i:0{len(str(MAX_EPOCHS))}}/{MAX_EPOCHS}:')
        
        model.train()
        optimizer.zero_grad()
        
        # forward pass
        output = model(features, adj)
        
        # compute the training loss and accuracy
        train_loss = criterion(output, labels)
        acc_train = accuracy(output, labels)
        
        # backpropogate the loss
        train_loss.backward()
        optimizer.step()
        
        # evaluate
        model.eval()
        output = model(features, adj)
        val_loss = criterion(output, labels)
        val_acc = accuracy(output, labels)
                
        results['epoch'].append(i)
        results['train_loss'].append(train_loss.item())        
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss.item())
        results['val_acc'].append(val_acc)
        print(results)
        
        # <---------------------------------------- THIS IS HOW FAR I GOT
        
        best_accuracy = torch.tensor(val_acc_fn).max().item()
        create_checkpoint(checkpoints_dir, epoch, model, optimizer, results, best_accuracy)
        if (epoch+1) % 4 == 0 and epoch != 0:
            learning_rate = learning_rate / 2
            optimizer = optim.Adam(
                    params=model.parameters(),
                    lr=learning_rate)


    # save and plot the results
    save_results(results_dir, results, model)
    save_model(models_dir, model)
    plot_results(results_dir, results, model)



def main():
    # print all flags
    print_flags(FLAGS)

    # start the timer
    start_time = time()

    # train the model
    train()

    # end the timer
    end_time = time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Done training in {minutes}:{seconds} minutes.')


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE_DEFAULT,
                        help='Input mode (i.e: glove, elmo or glove_and_elmo)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Path of directory where the data is stored')
    parser.add_argument('--checkpoints_dir', type=str, default=CHECKPOINTS_DIR_DEFAULT,
                        help='Path of directory to store / load checkpoints')
    parser.add_argument('--models_dir', type=str, default=MODELS_DIR_DEFAULT,
                        help='Path of directory to store / load models')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR_DEFAULT,
                        help='Path of directory to store results')
    parser.add_argument('--run_desc', type=str, default=RUN_DESC_DEFAULT,
                        help='Run description, used to generate the subdirectory title')
    FLAGS, unparsed = parser.parse_known_args()

    main()
