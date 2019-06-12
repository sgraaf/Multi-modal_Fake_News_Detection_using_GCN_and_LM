"""
NOTE
Due to some problems with the CUDA/GPU memory loads, 
we had to change some code in the torch library.
Namely, in the rnn.py and functional.py scripts in torch
there are some added cpu/cuda assignments. 
The code will work on both cpu and gpu. 
Functions pack_padded_sequence in rnn.py
and nll_loss in function.py 
were edited. 

Fix source: https://discuss.pytorch.org/t/error-with-lengths-in-pack-padded-sequence/35517/6
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator  # , Iterator
# from allennlp.modules.elmo import Elmo
import torch.utils.data as data
from torchtext.vocab import GloVe

#from data import load_data
from dataset import FNNDataset, PadSortBatchFNN
from models import HierarchicalAttentionNet
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_model, save_results, create_checkpoint, get_number_sentences,
                   get_class_balance)

# defaults
FLAGS = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

ROOT_DIR = Path.cwd().parent
LEARNING_RATE = 0.1
MAX_EPOCHS = 20
BATCH_SIZE_FN = 32
NUM_CLASSES_FN = 2

WORD_EMBED_DIM = 300
ELMO_EMBED_DIM = 1024
ELMO_EMBED_DIM = None
WORD_HIDDEN_DIM = 100
SENT_HIDDEN_DIM = 100


MODEL_TYPE_DEFAULT = 'glove_and_elmo'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
RUN_DESC_DEFAULT = None

# ELMO_DIR = Path().cwd().parent / 'data' / 'elmo'
# ELMO_OPTIONS_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
# ELMO_WEIGHT_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

    
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

    # get the glove and elmo embedding
    glove_dim = 0
    elmo_dim = 0
    GloVe_vectors = None
    ELMo = None
    if 'glove' in model_type:
        GloVe_vectors = GloVe()
        glove_dim = WORD_EMBED_DIM
        print('Uploaded GloVe embeddings.')
    if 'elmo' in model_type:
        ELMo = Elmo(
            options_file=ELMO_OPTIONS_FILE, 
            weight_file=ELMO_WEIGHT_FILE,
            num_output_representations=1, 
            requires_grad=False,
            dropout=0).to(DEVICE)
        elmo_dim = ELMO_EMBED_DIM
        print('Uploaded Elmo embeddings.')
    input_dim = glove_dim + elmo_dim
    # get the fnn and snli data
    FNN = {}
    FNN_DL = {}

    for path in ['train', 'val', 'test']:
        FNN[path] = FNNDataset(data_dir / ('FNN_' + path + '.pkl'), GloVe_vectors, ELMo)
        FNN_DL[path] = data.DataLoader(
                dataset=FNN[path],
                batch_size=BATCH_SIZE_FN,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                collate_fn=PadSortBatchFNN())
    print('Uploaded FNN data.')
    
    fnn_train_sent_no = get_number_sentences(data_dir / 'FNN_train.pkl')
    fnn_train_len = len(FNN['train'])


    # initialize the model, according to the model type
    print('Initializing the model...', end=' ')
    
    model = HierarchicalAttentionNet(input_dim=input_dim , 
                                     hidden_dim=WORD_HIDDEN_DIM, 
                                     num_classes=NUM_CLASSES_FN,  
                                     dropout=0).to(DEVICE)
    print('Working on: ', end='')
    print(DEVICE)
    print('Done!')
    print_model_parameters(model)
    print()

    # set the criterion and optimizer
    # we weigh the loss: class [0] is real, class [1] is fake
    # 
    real_ratio, fake_ratio = get_class_balance(data_dir / 'FNN_train.pkl')
    weights = [(1.0 - real_ratio), (1.0- fake_ratio)]
    print(weights)
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    loss_func_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE)        
    
    # load the last checkpoint (if it exists)
    epoch, results, best_accuracy = load_latest_checkpoint(checkpoints_dir, model, optimizer)
    results = {'epoch':[], 'train_loss':[], 'train_accuracy':[], 'val_loss': [], 'val_accuracy': []}
    if epoch == 0:
        print(f'Starting training at epoch {epoch + 1}...')
    else:
        print(f'Resuming training from epoch {epoch + 1}...')

    for i in range(epoch, MAX_EPOCHS):
        print(f'Epoch {i+1:0{len(str(MAX_EPOCHS))}}/{MAX_EPOCHS}:')
        model.train()
        # one epoch of training
        train_loss_fn, train_acc_fn = train_epoch_fn(FNN_DL['train'], model, 
                                                   optimizer, loss_func_fn)

        # one epoch of eval
        model.eval()
        val_loss_fn, val_acc_fn = eval_epoch_fn(FNN_DL['val'], model, 
                                              loss_func_fn)
        
        results['epoch'].append(i)
        results['train_loss'].append(train_loss_fn)        
        results['train_accuracy'].append(train_acc_fn)
        results['val_loss'].append(val_loss_fn)
        results['val_accuracy'].append(val_acc_fn)
        print(results)
        
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
