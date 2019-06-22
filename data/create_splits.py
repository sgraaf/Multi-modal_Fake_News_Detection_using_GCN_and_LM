#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:59:37 2019

@author: stevenvandegraaf
"""

import pickle as pkl
import random
from pathlib import Path

DATA_DIR = Path('/Users/stevenvandegraaf/stack/uva_vu_ai/project_ai_1/Multi-modal-Fake-News-Detection-using-GCN-and-LM/data/')
label_file = DATA_DIR / 'labels_matrix.pkl'

# load the labels list
labels_list = [label for label in pkl.load(open(label_file, 'rb'))]

# get idxs
idxs = [i for i in range(len(labels_list)) if labels_list[i]]
pos_idxs = [i for i in range(len(labels_list)) if labels_list[i] == [1, 0]]
neg_idxs = [i for i in range(len(labels_list)) if labels_list[i] == [0, 1]]

# shuffle the idxs
random.shuffle(pos_idxs)
random.shuffle(neg_idxs)

# get balanced splits
train_idxs = pos_idxs[:100] + neg_idxs[:100]
test_idxs = pos_idxs[100:200] + neg_idxs[100:200]
val_idxs = pos_idxs[200:] + neg_idxs[200:]

# create splits dict and pickle
split_idxs = {
    'train': train_idxs,
    'val': val_idxs,
    'test': test_idxs
}
splits_file = DATA_DIR / 'splits_dict.pkl'
pkl.dump(split_idxs, open(splits_file, 'wb'))
