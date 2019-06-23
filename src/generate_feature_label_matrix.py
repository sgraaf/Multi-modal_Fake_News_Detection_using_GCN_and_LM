import torch
from scipy import sparse
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from models import HierarchicalAttentionNet

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE_FN = 1
NUM_CLASSES_FN = 2

WORD_EMBED_DIM = 300
ELMO_EMBED_DIM = 1024
ELMO_EMBED_DIM = None
WORD_HIDDEN_DIM = 100
SENT_HIDDEN_DIM = 100

ADJ_DIR = '../adj_matrix/sparse_data1.npz'
MAP_ADJ_DIR = '../adj_matrix/adj_matrix_dict.txt'
DATA_DIR = '../fakenewsnet/'
NO_USERS = 35322


def generate_feature_label_matrix(adj_dir, map_adj_dir, data_dir, no_users=35322):

    # Loading the adjacency matrix and id-to-idx mapping dictionary
    adj_matrix = sparse.load_npz(adj_dir)
    adj_keys = {}
    with open(DICT_DIR, 'r') as f:
        for idx, x in enumerate(f):
            print(x)
            adj_keys[idx] = str(x).replace('\n','')
    inv_adj_keys = {v: k for k, v in adj_keys.items()}

    # Loading the ids of BF and PF articles
    bf_ids = {}
    with open(DATA_DIR + 'BuzzFeedNews.txt', 'r') as f:
        for idx, x in enumerate(f):
            bf_ids[str(x).replace('\n','')] = idx + 1 
    pf_ids = {}
    with open(DATA_DIR + 'PolitiFactNews.txt', 'r') as f:
        for idx, x in enumerate(f):
            pf_ids[str(x).replace('\n','')] = idx + 1

    bf_dirs = ['BuzzFeed_real_news_content.csv', 'BuzzFeed_fake_news_content.csv']
    pf_dirs = ['PolitiFact_real_news_content.csv', 'PolitiFact_fake_news_content.csv']

    # Uploading the article text data and labels
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

    # Initializing and fitting a binary BoW-vectorizer
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(all_texts)

    # Initializing the feature and label matrices
    feature_matrix = [None]*len(inv_adj_keys)
    label_matrix = [None]*len(inv_adj_keys)

    # Filling in the rows in feature and label matrices for ARTICLE nodes
    text_count = 0
    for outlet in texts:
        for article in texts[outlet]:
            id_map = outlet + str(article[0])
            id_mapped = int(inv_adj_keys[id_map])
            feature_matrix[id_mapped] = vectorizer.transform([article[1]])
            if feature_matrix[id_mapped] is not None:
                text_count += 1 
            if article[2] == 0:
                label_matrix[id_mapped] = [1,0]
            elif article[2] == 1:
                label_matrix[id_mapped] = [0,1]

    # Filling in the rows in feature and label matrices for USER nodes
    for idx in range(adj_matrix.shape[0]):
        temp_bow = None
        count_bow = 0
        for pair in adj_matrix.getrow(idx).nonzero():
            if pair[1] >= no_users:
                if temp_bow is None:
                    temp_bow = feature_matrix[pair[1]]
                else:
                    temp_bow += feature_matrix[pair[1]]
                count_bow += 1
        if temp_bow is None:
            feature_matrix[idx] = vectorizer.transform([''])
        else:
            feature_matrix[idx] = temp_bow
        if idx > no_users:
            break

    # Save files
    with open('feature_matrix.pkl', 'wb') as f:
        pickle.dump(feature_matrix,f)
    with open('label_matrix.pkl', 'wb') as f:
        pickle.dump(label_matrix,f)

if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    adj_dir, map_adj_dir, data_dir, no_users=35322
    parser.add_argument('--adj_dir', type=str, default=ADJ_DIR,
                        help='Path to file with adjacency matrix')
    parser.add_argument('--map_adj_dir', type=str, default=MAP_ADJ_DIR,
                        help='Path to file with adjacency matrix id mappings')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Path to directory with article and user data')
    parser.add_argument('--no_user', type=int, default=NO_USERS,
                        help='Number of users in adjacency matrix')
