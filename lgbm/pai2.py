import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# helper functions
def pickle_importer(path):
    file = open(path,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def column_normalizer(array):
    means = np.mean(array, axis=0)
    var = np.var(array, axis=0)
    array_n = (array - means)/np.sqrt(var)
    return array_n

def accuracy_calc(preds, true, cutoff):
    preds[preds >= cutoff] = 1
    preds[preds < cutoff] = 0
    accuracy = 1 - (np.sum(np.abs((preds - true))) / len(true))
    return accuracy

def avg_pool(list_):
    new_list = []
    for lists in list_:
        avg_pooled = np.zeros((1,16))
        for minilist in lists:
            user_embed = np.asarray(minilist)
            avg_pooled += user_embed / len(lists)
        new_list.append(avg_pooled)
    return new_list

# declare paths
PATH_GCN_TRAIN = '/Users/mnauman/Downloads/GCN_train_embeds_post_ReLU.pkl'
PATH_GCN_TEST = '/Users/mnauman/Downloads/GCN_test_embeds_post_ReLU.pkl'
PATH_GCN_VALID = '/Users/mnauman/Downloads/GCN_val_embeds_post_ReLU.pkl'
PATH3 = '/Users/mnauman/Downloads/FNN_small_test.pkl'
PATH4 = '/Users/mnauman/Downloads/FNN_small_train.pkl'
PATH5 = '/Users/mnauman/Downloads/FNN_small_val.pkl'
PATH_GLOVE_TRAIN = '/Users/mnauman/Downloads/FNN_small_embeds_glove_reduced_dim_glove_train (1).pkl'
PATH_GLOVE_TEST = '/Users/mnauman/Downloads/FNN_small_embeds_glove_reduced_dim_glove_test (1).pkl'
PATH_GLOVE_VALID = '/Users/mnauman/Downloads/FNN_small_embeds_glove_reduced_dim_glove_val (1).pkl'
PATH_GCN_USER_TRAIN = '/Users/mnauman/Downloads/GCN_train_user_embeds_post_ReLU.pkl'
PATH_GCN_USER_TEST = '/Users/mnauman/Downloads/GCN_test_user_embeds_post_ReLU.pkl'
PATH_GCN_USER_VALID = '/Users/mnauman/Downloads/GCN_val_user_embeds_post_ReLU.pkl'

# load labels
labels_test = pickle_importer(PATH3)
labels_train = pickle_importer(PATH4)
labels_valid = pickle_importer(PATH5)
labels_test = labels_test['labels']
labels_train = labels_train['labels']
labels_valid = labels_valid['labels']

# load GCN
gcn_train = pickle_importer(PATH_GCN_TRAIN)
gcn_test = pickle_importer(PATH_GCN_TEST)
gcn_valid = pickle_importer(PATH_GCN_VALID)

# load embeddings
glove_train = pickle_importer(PATH_GLOVE_TRAIN)
glove_test = pickle_importer(PATH_GLOVE_TEST)
glove_valid = pickle_importer(PATH_GLOVE_VALID)

# load user embeddings
user_train = pickle_importer(PATH_GCN_USER_TRAIN)
user_test = pickle_importer(PATH_GCN_USER_TEST)
user_valid = pickle_importer(PATH_GCN_USER_VALID)

# average_pooling
user_train = avg_pool(user_train)
user_test = avg_pool(user_test)
user_valid = avg_pool(user_valid)

#prepare the data
for i in range(22):
    gcn_train.append(gcn_valid[i])
    labels_train.append(labels_valid[i])
    glove_train.append(glove_valid[i])
    user_train.append(user_valid[i])
    
del i, labels_valid, gcn_valid, glove_valid, user_valid

gcn_train = np.asarray(gcn_train)
labels_train = np.asarray(labels_train)
gcn_test = np.asarray(gcn_test)
labels_test = np.asarray(labels_test)
glove_train = np.asarray(glove_train)
glove_test = np.asarray(glove_test)
user_test = np.asarray(user_test).reshape(200,16)
user_train = np.asarray(user_train).reshape(222, 16)

normalized = np.zeros((422, 16+32))
normalized[:200,:16] = gcn_test
normalized[200:,:16] = gcn_train
normalized[:200,16:] = glove_test
normalized[200:,16:] = glove_train

normalized2 = np.zeros((422, 16))
normalized2[:200,:] = user_test
normalized2[200:,:] = user_train

normalized = np.c_[normalized, normalized2]

normalized = column_normalizer(normalized)
normalized = normalized[:,~np.all(np.isnan(normalized), axis=0)]
test = normalized[:200,:]
train = normalized[200:,:]

# declare parameters
parameters = {'num_trees':1000,
              'max_depth':2,
              'min_data_in_leaf':25,
              'objective':'xentropy',
              'boosting':'gbdt',  
              'device_type':'cpu',
              'learning_rate':0.1,
              'boost_from_average':'False'}   

# train lgbm
def lgb_runner(train, target, test, target_test, parameters, seed=1, number_of_folds=10):
    validation_preds = np.zeros((len(train)))
    train_preds = np.zeros((len(train)))
    test_preds = np.zeros((len(test)))
    kfold = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
    for train_fold, valid_fold in kfold.split(train, target):
        train_data = lgb.Dataset(train[train_fold], label=target[train_fold])
        validation_data = lgb.Dataset(train[valid_fold], label = target[valid_fold], reference=train_data)
        model = lgb.train(parameters, train_data, valid_sets=[validation_data, train_data], early_stopping_rounds = 100)
        validation_preds[valid_fold] = model.predict(train[valid_fold], num_iteration=model.best_iteration)
        train_preds[train_fold] += model.predict(train[train_fold], num_iteration=model.best_iteration)/(number_of_folds-1)
        test_preds += model.predict(test, num_iteration=model.best_iteration)/number_of_folds
    return validation_preds, train_preds, test_preds

a, aa, aaa = lgb_runner(labels_train_gcn, labels_train, labels_test_gcn, labels_test, parameters, seed=1, number_of_folds=10)

# train logreg
def logreg_runner(train, target, test, target_test, parameters, seed):
    train_preds = np.zeros((len(train)))
    test_preds = np.zeros((len(test)))
    model = LogisticRegression(penalty=parameters['penalty'], dual=False, tol=0.0001, 
                                   C=parameters['c'], fit_intercept=True, intercept_scaling=1, 
                                   random_state=seed, solver=parameters['solver'], 
                                   max_iter=5000, verbose=0).fit(train, target)     
    train_preds = model.predict(train)
    test_preds = model.predict(test)
    return train_preds, test_preds

def log_reg_looper(train, target, test, target_test, parameter_dict):
    accuracies = np.zeros((len(parameter_dict), 2))
    for i in range(len(parameter_dict)):
        train_preds, test_preds = logreg_runner(train, target, test, target_test, parameter_dict[i], i)
        accuracies[i,0] = accuracy_calc(train_preds, labels_train, 0.5)
        accuracies[i,1] = accuracy_calc(test_preds, labels_test, 0.5)
    return accuracies

def parameter_generator(n):
    parameter_dict = []
    for i in range(n):
        parameters = dict()
        parameters['solver'] = np.random.choice(['newton-cg', 'lbfgs', 'liblinear'])
        parameters['c'] = np.random.choice([1, 0.1, 0.01, 0.001, 0.0001,])
        if parameters['solver'] == 'liblinear':
            parameters['penalty'] = np.random.choice(['l1', 'l2'])
        else:
            parameters['penalty'] = 'l2'
        parameter_dict.append(parameters)
    return parameter_dict

# train nn
def nn_runner(parameters, train, target, valid, seed):
    train_preds = np.zeros((len(train)))
    valid_preds = np.zeros((len(test)))
    model = MLPClassifier(activation='tanh', alpha=parameters[0], early_stopping=True, hidden_layer_sizes=parameters[1],
              learning_rate='adaptive', learning_rate_init=0.01,
              max_iter=2000, n_iter_no_change=100, random_state=seed,
              shuffle=True, solver='adam', validation_fraction=0.1).fit(train, target)        
    train_preds = model.predict(train)
    valid_preds = model.predict(valid)
    return train_preds, valid_preds

def nn_optimizer(params, train, target, valid, valid_target):
    accuracies = np.zeros((len(params), 2))
    for i in range(len(params)):
        train_preds, valid_preds = nn_runner(params[i], train, target, valid, i)
        accuracies[i,0] = accuracy_calc(train_preds, target, 0.5)
        accuracies[i,1] = accuracy_calc(valid_preds, valid_target, 0.5)
    return accuracies
        

# train logreg
def logreg_runner2(number_of_folds, train, target, seed):
    train_preds = np.zeros((len(train)))
    valid_preds = np.zeros((len(train)))
    kfold = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
    i = 0
    for train_fold, valid_fold in kfold.split(train, target):
        model = LogisticRegression(penalty=parameters[i]['penalty'], dual=False, tol=0.0001, 
                                   C=parameters[i]['c'], fit_intercept=True, intercept_scaling=1, 
                                   random_state=seed, solver=parameters[i]['solver'], 
                                   max_iter=5000, verbose=0).fit(train[train_fold], target[train_fold])          
        train_preds[train_fold] += model.predict(train[train_fold])/(number_of_folds-1)
        valid_preds[valid_fold] = model.predict(train[valid_fold])
        i += 1
    return train_preds, test_preds






