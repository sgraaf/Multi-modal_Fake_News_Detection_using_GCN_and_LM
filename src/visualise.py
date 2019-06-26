import csv
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse

def visualize(results_dir):
    results_file = results_dir + 'HierarchicalAttentionNet_results.csv'

    df = pd.read_csv(results_file, sep=';')
    print(df.keys())
            
    temp = []
    for i in df['train_accuracy']:
        i = ast.literal_eval(i.replace('tensor','').replace('(','').replace(')',''))
        temp.append(i)
    df['train_accuracy'] = temp
    
    temp = []
    for i in df['val_accuracy']:
        i = ast.literal_eval(i.replace('tensor','').replace('(','').replace(')',''))
        temp.append(i)
    df['val_accuracy'] = temp


    # Plotting train accuracy
    fn_tr_acc_mean = []
    for i in df['train_accuracy']:
        for j in i:
            fn_tr_acc_mean.append(sum(i) / len(i))
    fn_tr_acc = sum(df['train_accuracy'],[])
    plt.plot(fn_tr_acc, alpha=0.5)
    plt.plot(fn_tr_acc_mean)
    plt.legend(['per batch','mean per epoch'])
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('batch')
    plt.title('Accuracy on training set over the training process')
    plt.savefig(results_dir + 'train_acc.png')
    plt.show()

    # Plotting train loss
    plt.plot(df['train_loss'])
    plt.legend(['mean per epoch'])
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(min(range(len(df['train_loss']))), max(range(len(df['train_loss'])))+1, 1.0))
    plt.title('Loss on training set over the training process')
    plt.savefig(results_dir + 'train_loss.png')
    plt.show()

    # Plotting validation accuracy
    fn_val_acc_mean = []
    for i in df['val_accuracy']:
        for j in i:
            fn_val_acc_mean.append(sum(i) / len(i))
    fn_val_acc = sum(df['val_accuracy'],[])
    plt.plot(fn_val_acc, alpha=0.6)
    plt.plot(fn_val_acc_mean)
    plt.legend(['per batch','mean per epoch'])
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('batch')
    plt.title('Accuracy on validation set over the training process')
    plt.savefig(results_dir +'val_acc.png')
    plt.show()

    # Plotting validation loss    
    plt.plot(df['val_loss'])
    plt.legend(['mean per epoch'])
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(min(range(len(df['val_loss']))), max(range(len(df['val_loss'])))+1, 1.0))
    plt.title('Loss on validation set over the training process')
    plt.savefig(results_dir + 'val_loss.png')
    plt.show()
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizing HAN performance results.')
    parser.add_argument('--results_dir', dest='results_dir', help='path to folder with results file', default="./")
    args = parser.parse_args()

    results_dir = args.results_dir
    visualize(results_dir)
