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

SPARSE_DIR = '../adj_matrix/sparse_data1.npz'
DICT_DIR = '../adj_matrix/adj_matrix_dict.txt'
DATA_DIR = '../fakenewsnet/'



sparse_matrix = sparse.load_npz(SPARSE_DIR)
sparse_keys = {}
with open(DICT_DIR, 'r') as f:
    for idx, x in enumerate(f):
        sparse_keys[idx] = str(x).replace('\n','')
inv_sparse_keys = {v: k for k, v in sparse_keys.items()}

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

#print(len(texts['buzz']))
#print(len(texts['pol']))
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(all_texts)

feature_matrix = [None]*len(inv_sparse_keys)
label_matrix = [None]*len(inv_sparse_keys)

no_users = 35322



text_count = 0
for outlet in texts:
    for article in texts[outlet]:
        id_map = outlet + str(article[0])
        id_mapped = int(inv_sparse_keys[id_map])
        feature_matrix[id_mapped] = vectorizer.transform([article[1]])
        if feature_matrix[id_mapped] is not None:
            text_count += 1 
        if article[2] == 0:
            label_matrix[id_mapped] = [1,0]
        elif article[2] == 1:
            label_matrix[id_mapped] = [0,1]


for idx in range(sparse_matrix.shape[0]):
    temp_bow = None
    count_bow = 0
    for pair in sparse_matrix.getrow(idx).nonzero():
        if pair[1] >= no_users:
            if temp_bow is None:
                temp_bow = feature_matrix[pair[1]]
            else:
                temp_bow += feature_matrix[pair[1]]
            count_bow += 1
    if temp_bow is None:
        feature_matrix[idx] = vectorizer.transform([''])
        #print(feature_matrix[idx])
    else:
        feature_matrix[idx] = temp_bow
    if idx > no_users:
        break

with open('feature_matrix.pkl', 'wb') as f:
    pickle.dump(feature_matrix,f)
with open('label_matrix.pkl', 'wb') as f:
    pickle.dump(label_matrix,f)








"""
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

# get the fnn

FNN_small = FNNDataset(data_dir / ('FNN_small.pkl'), GloVe_vectors, ELMo)
batch_size = len(FNN_small)
FNN_DL_small = data.DataLoader(
        dataset=FNN_small,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        collate_fn=PadSortBatchFNN())
print('Uploaded FNN data.')


model = initialize_han(input_dim, WORD_HIDDEN_DIM, NUM_CLASSES_FN, DEVICE)
load_model(model_path, model)
for step, batch in enumerate(FNN_DL_small):       
    doc_embeds = get_article_embeddings(model, batch)
"""

def load_model(model_path, model):
    """
    Loads a checkpoint

    :param pathlib.Path checkpoint_path: the path of the checkpoint
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :returns: tuple of epoch, model, optimizer, results and best_accuracy of the checkpoint
    :rtype: tuple(int, nn.Module, optim.Optimizer, dict, float)
    """
    print('Loading the model...', end=' ')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Done!')

def initialize_han(input_dim, word_hidden_dim, num_classes_fn, device):
    """
    Initializes a Hierarchical Attention Network.

    :param input_dim: Dimensionality of word embeddings
    :param word_hidden_dim: Dimensionality of word hidden dimension 
    :param num_classes_fn: Number of labels in the dataset
    :param device: CUDA or CPU
    :returns: defined HAN model
    """
    model = HierarchicalAttentionNet(input_dim=input_dim , 
        hidden_dim=word_hidden_dim, 
        num_classes=num_classes_fn,  
        dropout=0).to(device)
    
    return model

def get_article_embeddings(model, batch):
    """
    Get article embeddings given a batch.

    :param model: pretrained HAN model
    :param batch: batch of articles
    :returns: article embeddings
    """
    articles, article_dims, labels = batch
    model.eval()
    doc_embeds = model.get_article_embeddings(batch=articles,
        batch_dims=article_dims,
        return_docs=True)

    return doc_embeds
