







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
