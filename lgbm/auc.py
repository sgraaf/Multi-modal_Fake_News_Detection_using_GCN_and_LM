#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:17:46 2019

@author: mnauman
"""
import sklearn.metrics as metrics
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

def softmax(x):
    exp = np.exp(x)
    row_sum = np.sum(exp, axis = 1).reshape(len(x), 1)
    sm = exp / row_sum
    return sm

# only difference

# import the data
    
PATH_GCN_TRAIN = '/Users/mnauman/Downloads/GCN_train_outputs.pkl'
PATH_GCN_TEST = '/Users/mnauman/Downloads/GCN_test_outputs.pkl'
PATH_GCN_VALID = '/Users/mnauman/Downloads/GCN_val_outputs.pkl'
PATH3 = '/Users/mnauman/Downloads/FNN_small_test.pkl'
PATH4 = '/Users/mnauman/Downloads/FNN_small_train.pkl'
PATH5 = '/Users/mnauman/Downloads/FNN_small_val.pkl'
PATH_GLOVE_VALID = '/Users/mnauman/Downloads/FNN_small_softmax_only_glove_val.pkl'
PATH_GLOVE_TRAIN = '/Users/mnauman/Downloads/FNN_small_softmax_only_glove_train.pkl'
PATH_GLOVE_TEST = '/Users/mnauman/Downloads/FNN_small_softmax_only_glove_test.pkl'
PATH_GLOVE_ELMO_VALID = '/Users/mnauman/Downloads/FNN_small_softmax_glove_and_elmo_val.pkl'
PATH_GLOVE_ELMO_TRAIN = '/Users/mnauman/Downloads/FNN_small_softmax_glove_and_elmo_train.pkl'
PATH_GLOVE_ELMO_TEST = '/Users/mnauman/Downloads/FNN_small_softmax_glove_and_elmo_test.pkl'
PATH_G_E_TL_VALID = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove__elmo_0001_val.pkl'
PATH_G_E_TL_TRAIN = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove__elmo_0001_train.pkl'
PATH_G_E_TL_TEST = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove__elmo_0001_test.pkl'
PATH_G_TL_VALID = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove_0001_val.pkl'
PATH_G_TL_TRAIN = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove_0001_train.pkl'
PATH_G_TL_TEST = '/Users/mnauman/Downloads/FNN_small_softmax_tl_glove_0001_test.pkl'

gcn_test = pickle_importer(PATH_GCN_TEST)
gcn_train = pickle_importer(PATH_GCN_TRAIN)
gcn_valid = pickle_importer(PATH_GCN_VALID)

glove_elmo_valid = pickle_importer(PATH_GLOVE_ELMO_VALID)
glove_elmo_train = pickle_importer(PATH_GLOVE_ELMO_TRAIN)
glove_elmo_test = pickle_importer(PATH_GLOVE_ELMO_TEST)

labels_test = pickle_importer(PATH3)
labels_train = pickle_importer(PATH4)
labels_valid = pickle_importer(PATH5)
labels_test = labels_test['labels']
labels_train = labels_train['labels']
labels_valid = labels_valid['labels']

glove_valid = pickle_importer(PATH_GLOVE_VALID)
glove_train = pickle_importer(PATH_GLOVE_TRAIN)
glove_test = pickle_importer(PATH_GLOVE_TEST)

g_valid = pickle_importer(PATH_G_TL_VALID)
g_train = pickle_importer(PATH_G_TL_TRAIN)
g_test = pickle_importer(PATH_G_TL_TEST) 

g_e_valid = pickle_importer(PATH_G_E_TL_VALID)
g_e_train = pickle_importer(PATH_G_E_TL_TRAIN)
g_e_test = pickle_importer(PATH_G_E_TL_TEST)
# preprocess data

gcn_valid = np.exp(np.asarray(gcn_valid))
labels_valid = np.asarray(labels_valid)
gcn_train = np.exp(np.asarray(gcn_train))
labels_train = np.asarray(labels_train)
gcn_test = np.exp(np.asarray(gcn_test))
labels_test = np.asarray(labels_test)
glove_train = softmax(np.asarray(glove_train))
glove_test = softmax(np.asarray(glove_test))
glove_valid = softmax(np.asarray(glove_valid))
glove_elmo_train = softmax(np.asarray(glove_elmo_train))
glove_elmo_test = softmax(np.asarray(glove_elmo_test))
glove_elmo_valid = softmax(np.asarray(glove_elmo_valid))
g_test = softmax(np.asarray(g_test))
g_e_test = softmax(np.asarray(g_e_test))
g_train = softmax(np.asarray(g_train))
g_e_train = softmax(np.asarray(g_e_train))
g_valid = softmax(np.asarray(g_valid))
g_e_valid = softmax(np.asarray(g_e_valid))


train = np.zeros((200,3))
test = np.zeros((200,3))
valid = np.zeros((22,3))

train[:,0] = gcn_train[:,1]
train[:,1] = g_train[:,1]
train[:,2] = g_e_train[:,1]

test[:,0] = gcn_test[:,1]
test[:,1] = g_test[:,1]
test[:,2] = g_e_test[:,1]

valid[:,0] = gcn_valid[:,1]
valid[:,1] = g_valid[:,1]
valid[:,2] = g_e_valid[:,1]



# ensemble

prediction_train = np.zeros((200,6))
#prediction_valid = np.zeros((22,6))
#prediction_test = np.zeros((200,7))

#prediction_train[:,0] = glove_train[:,1]
#prediction_train[:,1] = gcn_train[:,1]
#prediction_train[:,2] = np.mean(train, axis=1)
#prediction_train[:,3] = np.mean(train2, axis=1)

#prediction_valid[:,0] = glove_valid[:,1]
#prediction_valid[:,1] = gcn_valid[:,1]
#prediction_valid[:,2] = np.mean(valid, axis=1)
#prediction_valid[:,3] = np.mean(valid2, axis=1)

prediction_test[:,0] = glove_test[:,1]
prediction_test[:,1] = glove_elmo_test[:,1]
prediction_test[:,2] = gcn_test[:,1]
prediction_test[:,3] = g_test[:,1]
prediction_test[:,4] = g_e_test[:,1]
prediction_test[:,5] = (g_test[:,1] + gcn_test[:,1])/2
prediction_test[:,6] = (g_e_test[:,1] + gcn_test[:,1])/2

accs = np.zeros(7)
for i in range(7):
    accs[i] = accuracy_calc(prediction_test[:,i], labels_test, 0.5)
    

fpr, tpr, threshold = metrics.roc_curve(labels_test, prediction_test[:,0])
roc_auc = metrics.auc(fpr, tpr)

fpr2, tpr2, threshold2 = metrics.roc_curve(labels_test, prediction_test[:,1])
roc_auc2 = metrics.auc(fpr2, tpr2)

fpr3, tpr3, threshold3 = metrics.roc_curve(labels_test, prediction_test[:,2])
roc_auc3 = metrics.auc(fpr3, tpr3)

fpr4, tpr4, threshold4 = metrics.roc_curve(labels_test, prediction_test[:,3])
roc_auc4 = metrics.auc(fpr4, tpr4)

fpr5, tpr5, threshold5 = metrics.roc_curve(labels_test, prediction_test[:,4])
roc_auc5 = metrics.auc(fpr5, tpr5)

fpr6, tpr6, threshold6 = metrics.roc_curve(labels_test, prediction_test[:,5])
roc_auc6 = metrics.auc(fpr6, tpr6)

fpr7, tpr7, threshold7 = metrics.roc_curve(labels_test, prediction_test[:,6])
roc_auc7 = metrics.auc(fpr7, tpr7)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.figure(figsize=(20,20))
plt.plot(fpr, tpr, label = 'HAN (GloVe); AUC = %0.2f' % roc_auc)
plt.plot(fpr2, tpr2, label = 'HAN (GloVe/ELMo); AUC = %0.2f' % roc_auc2)
plt.plot(fpr4, tpr4, label = 'HAN  (TL; GloVe); AUC = %0.2f' % roc_auc4)
plt.plot(fpr5, tpr5, label = 'HAN (TL; GloVe/ELMo); AUC = %0.2f' % roc_auc5)
plt.plot(fpr3, tpr3, label = 'GCN; AUC = %0.2f' % roc_auc3)
plt.plot(fpr6, tpr6, label = 'GCN + HAN 1; AUC = %0.2f' % roc_auc6)
plt.plot(fpr7, tpr7, label = 'GCN + HAN 2; AUC = %0.2f' % roc_auc7)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
