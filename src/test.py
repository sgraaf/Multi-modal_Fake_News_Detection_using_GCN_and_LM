
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time
import math
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator  # , Iterator
from allennlp.modules.elmo import Elmo
import torch.utils.data as data
from torchtext.vocab import GloVe

#from data import load_data
from dataset import FNNDataset, PadSortBatchFNN
from models import HierarchicalAttentionNet
from utils import (create_directories, load_latest_checkpoint, plot_results,
                   print_dataset_sizes, print_flags, print_model_parameters,
                   save_model, save_results, create_checkpoint, get_number_sentences,
                   get_class_balance, load_checkpoint)

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
BATCH_SIZE_FN = 1
NUM_CLASSES_FN = 2

WORD_EMBED_DIM = 300
ELMO_EMBED_DIM = 1024
SENT_HIDDEN_DIM = 100
DOC_HIDDEN_DIM = 100


MODEL_TYPE_DEFAULT = 'glove_and_elmo'
DATA_DIR_DEFAULT = ROOT_DIR / 'data'
CHECKPOINTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_DIR_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_DIR_DEFAULT = ROOT_DIR / 'output' / 'results'
RUN_DESC_DEFAULT = None

ELMO_DIR = Path().cwd().parent / 'data' / 'elmo'
ELMO_OPTIONS_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
ELMO_WEIGHT_FILE = ELMO_DIR / 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

def test():
    model_type = FLAGS.model_type
    run_desc = FLAGS.run_desc
    data_dir = Path(FLAGS.data_dir)
    checkpoints_dir = Path(FLAGS.checkpoints_dir) / model_type / run_desc
    models_dir = Path(FLAGS.models_dir) / model_type / run_desc
    results_dir = Path(FLAGS.results_dir) / model_type / run_desc
    learning_rate = LEARNING_RATE
    epoch_no = FLAGS.epoch
    sent_hidden_dim = FLAGS.sent_hidden_dim
    doc_hidden_dim = FLAGS.doc_hidden_dim

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

    FNN_test = FNNDataset(data_dir / ('FNN_test.pkl'), GloVe_vectors, ELMo)
    FNN_DL_test = data.DataLoader(
                dataset=FNN_test,
                batch_size=BATCH_SIZE_FN,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                collate_fn=PadSortBatchFNN())
    print('Uploaded FNN data.')

    print('Initializing the model...', end=' ')
    
    model = HierarchicalAttentionNet(input_dim=input_dim , 
                                     sent_hidden_dim=sent_hidden_dim,
                                     doc_hidden_dim=doc_hidden_dim, 
                                     num_classes=NUM_CLASSES_FN,  
                                     dropout=0).to(DEVICE)
    print('Working on: ', end='')
    print(DEVICE)
    print('Done!')
    print_model_parameters(model)
    print()

    optimizer = optim.Adam(
            params=model.parameters(),
            lr=LEARNING_RATE)        

    #model.load_state_dict(torch.load(CHECKPOINTS_DIR_DEFAULT / 'HierarchicalAttentionNet_model.pt'))
    if epoch_no == '0':
        model_path = models_dir / Path('HierarchicalAttentionNet_model.pt')
        _, _, _ = load_latest_checkpoint(model_path, model, optimizer)
    else:
        checkpoint_path = checkpoints_dir / Path('HierarchicalAttentionNet_Adam_checkpoint_' + str(epoch_no) + '_.pt')
        _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)
    model.eval()
    loss_func_fn = nn.CrossEntropyLoss()
    y_pred = []
    y_true = []
    for step, batch in enumerate(FNN_DL_test):       
        articles, article_dims, labels = batch
        out = model(batch=articles, batch_dims=article_dims)
        y_pred.append(out.argmax(dim=1).to(DEVICE).item())
        y_true.append(labels.to(DEVICE).item())
        #if step % 100 == 0 and step != 0:
            #print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro'))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro'))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None))

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
    parser.add_argument('--epoch', type=str, default=None,
                        help='Which epoch to select')
    parser.add_argument('--sent_hidden_dim', type=int, default=SENT_HIDDEN_DIM,
                        help='Dimensionality of sentence embeddings')
    parser.add_argument('--doc_hidden_dim', type=int, default=DOC_HIDDEN_DIM,
                        help='Dimensionality of document embeddings')

    FLAGS, unparsed = parser.parse_known_args()

    test()


