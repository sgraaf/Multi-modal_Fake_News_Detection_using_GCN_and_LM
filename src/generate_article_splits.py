#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:59:37 2019

@author: stevenvandegraaf
"""

import pickle as pkl
import random
from pathlib import Path
from nltk import tokenize
import pandas as pd
import csv

DATA_DIR = Path('../data/')
ARTICLE_DIR = Path('../fakenewsnet/')
ADJ_DIR = Path('../adj_matrix/')
splits_file = DATA_DIR / 'splits_dict.pkl'
map_file = ADJ_DIR / 'adj_matrix_dict.txt'

adj_keys = {}
with open(map_file, 'r') as f:
	for idx, x in enumerate(f):
		adj_keys[idx] = str(x).replace('\n','')
inv_adj_keys = {v: k for k, v in adj_keys.items()}

with open(splits_file, 'rb') as f:
	splits_dict = pkl.load(f)
# print(splits_dict)

bf_dirs = ['BuzzFeed_real_news_content.csv', 'BuzzFeed_fake_news_content.csv']
pf_dirs = ['PolitiFact_real_news_content.csv', 'PolitiFact_fake_news_content.csv']

bf_ids = {}
with open(ARTICLE_DIR / 'BuzzFeedNews.txt', 'r') as f:
	for idx, x in enumerate(f):
		bf_ids[str(x).replace('\n','')] = idx + 1 
pf_ids = {}
with open(ARTICLE_DIR / 'PolitiFactNews.txt', 'r') as f:
	for idx, x in enumerate(f):
		pf_ids[str(x).replace('\n','')] = idx + 1


dataset = {
	'articles': [],
	'labels': [],
	'idx': []
	}

for id_outlet, text_outlet in enumerate([bf_dirs, pf_dirs]):
	for dir_idx, dir_path in enumerate(text_outlet):
		with open(ARTICLE_DIR / dir_path, 'r') as f:
			reader = csv.reader(f)
			label = dir_idx
			for index, row in enumerate(reader):
				if index == 0:
					continue
				title = row[1].strip()
				text = row[2].strip()
				# tokenize the title + text into sentences
				article_sentences = tokenize.sent_tokenize(title) + tokenize.sent_tokenize(text)
				# tokenize the sentences into words
				article_sentences_words = [tokenize.word_tokenize(sentence) for sentence in article_sentences]

				# append the article and label
				dataset['articles'].append(article_sentences_words)
				dataset['labels'].append(label)
				if id_outlet == 0:
					idx = 'buzz' + str(bf_ids['BuzzFeed_' + str(row[0]).replace('\n','').replace('-Webpage','')])
				elif id_outlet == 1:
					idx = 'pol' + str(pf_ids['PolitiFact_' + str(row[0]).replace('\n','').replace('-Webpage','')])
				dataset['idx'].append(idx)
splits ={'train':{'articles': [], 'labels': []},
	'val':{'articles': [], 'labels': []},
	'test':{'articles': [], 'labels': []}}

for i in range(len(dataset['articles'])):
	for key in splits.keys():		
		mapped_id = int(inv_adj_keys[dataset['idx'][i]])		
		if mapped_id in splits_dict[key]:
			splits[key]['articles'].append(dataset['articles'][i])
			splits[key]['labels'].append(dataset['labels'][i])

for key in splits.keys():
	FNN_small_dir = ARTICLE_DIR / ('FNN_small_' + key + '.pkl')
	pkl.dump(splits[key], open(FNN_small_dir, 'wb'))




"""
texts = {'buzz':[], 'pol':[]}
all_texts = []
for id_outlet, text_outlet in enumerate([bf_dirs, pf_dirs]):
temp_texts = []    
for dir_idx, dir_path in enumerate(text_outlet):
label = dir_idx
with open(DATA_DIR + dir_path, 'r') as f:
reader = csv.reader(f)
for idx, x in enumerate(reader):
if idx == 0:
continue
if id_outlet == 0:
idx = bf_ids['BuzzFeed_' + str(x[0]).replace('\n','').replace('-Webpage','')]
elif id_outlet == 1:
idx = pf_ids['PolitiFact_' + str(x[0]).replace('\n','').replace('-Webpage','')]
temp_texts.append([idx, x[1] + ' ' + x[2], label])
all_texts.append(x[1] + str(x[2].split('\n')[0]))
if id_outlet == 0:
texts['buzz'] = temp_texts.copy()
else:
texts['pol'] = temp_texts.copy()
"""