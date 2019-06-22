import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import random

path = 'path'

#import the data
train = pd.read_csv(path)

def lgb_runner(folds, num_folds, parameters, train, target):
    validation_preds = np.zeros((len(train)))
    train_preds = np.zeros((len(train)))
    total_preds = np.zeros((len(train)))
    for fold, (training, validation) in enumerate(folds.split(train.values, target.values)):
        print('Working on fold number: ', fold)
        x_train = train.iloc[training]
        y_train = target.iloc[training]
        x_test = train.iloc[validation]
        y_test = target.iloc[validation]
        training_data = lgb.Dataset(x_train, label=y_train)
        validation_data = lgb.Dataset(x_test, label=y_test)
        model = lgb.train(parameters, training_data, valid_sets = [validation_data], early_stopping_rounds = 1000)
        validation_preds[validation] = model.predict(train.iloc[validation], num_iteration=model.best_iteration)
        train_preds[training] += model.predict(train.iloc[training], num_iteration=model.best_iteration) / (num_folds-1)
        total_preds += model.predict(train, num_iteration=model.best_iteration) / num_folds
        loss_train = 
        loss_val = 
        return loss_train, loss_val
    
def parameter_search(parameter_list, num_folds, train, target):
    loss = np.zeros((len(parameter_list), 3))
    for i in range(len(parameter_list)):
        folds = KFold(n_splits=num_folds, random_state=i)
        loss[i,0], loss[i,1] = lgb_runner(folds, num_folds, parameter_list[i], train, target)
    return loss

def parameter_generator(number_of_sets):
    parameter_list = []
    for i in range(number_of_sets):
        parameters = {
              'num_leaves':random.randint(),
              'num_trees':random.randint(),
              'max_depth':random.randint(-1,10),
              'min_data_in_leaf':random.randint(5,25),
              'min_sum_hessian_in_leaf':np.random.uniform(1e-4,1e-2),
              'bagging_fraction':np.random.uniform(0.01,0.99),
              'feature_fraction':np.random.uniform(0,1),
              'feature_fraction_seed':i,
              'bagging_seed':i,
              'objective':'',
              'device_type':'cpu',
              'learning_rate':np.random.uniform(0.001, 0.01),
              'boost_from_average':'True',
              'metric':''}
        parameter_list.append(parameters)
    return parameter_list
