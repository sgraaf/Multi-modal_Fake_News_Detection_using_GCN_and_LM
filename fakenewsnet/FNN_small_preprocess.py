#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
from pathlib import Path

from nltk import tokenize
import pandas as pd

DATASET_DIR = Path('/Users/stevenvandegraaf/stack/uva_vu_ai/project_ai_1/Multi-modal-Fake-News-Detection-using-GCN-and-LM/fakenewsnet')
CSV_files = sorted(DATASET_DIR.glob('*_news_content.csv'))

dataset = {
    'articles': [],
    'labels': []
}

for CSV_file in CSV_files:
    df = pd.read_csv(CSV_file, encoding='utf-8')
    
    if '_real_' in CSV_file.stem:
        label = 0
    else:
        label = 1 
        
    for index, row in df.iterrows():
        title = row['title'].strip()
        text = row['text'].strip()
        
        # tokenize the title + text into sentences
        article_sentences = tokenize.sent_tokenize(title) + tokenize.sent_tokenize(text)
        # tokenize the sentences into words
        article_sentences_words = [tokenize.word_tokenize(sentence) for sentence in article_sentences]

        # append the article and label
        dataset['articles'].append(article_sentences_words)
        dataset['labels'].append(label)

# pickle the dataset
FNN_small_file = DATASET_DIR / 'FNN_small.pkl'
pkl.dump(dataset, open(FNN_small_file, 'wb'))