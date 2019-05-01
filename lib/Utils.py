# solve the path problems
import sys, os
dirname = os.path.dirname(__file__)
sys.path.insert(0,dirname)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
import collections
import itertools
from lib import Dataset
import torch
from CustomDict import CustomDict

def get_evaluation_matrices(labels, probs):
    """
    Calculate the evaluation matrices such as accuracy.
    
    params labels: ground truth, 0 and 1
    params preds: prediction probabilities, should be a 1D array of floats that represent the positive class probs
    """
    
    # calculate the prediction (based on prob cutoff = 0.5
    preds = (np.array(probs)>0.5)*1

    # get the confusion matrix
    labels = np.array(labels)
    con_m = confusion_matrix(labels,preds)

    # ----- 3. Get tp, etc from confusion matrix
    tp = con_m[1,1]
    fn = con_m[1,0]
    tn = con_m[0,0]
    fp = con_m[0,1]

    print('tp: %s'%tp)
    print('fn: %s'%fn)
    print('tn: %s'%tn)
    print('fp: %s'%fp)
    # ----- 4. Get the evaluation martices
    matrix = collections.OrderedDict()
    matrix['accuracy'] = (tp+tn)/(tp+tn+fn+fp)
    matrix['precision'] = tp/(tp+fp+sys.float_info.epsilon)
    matrix['recall'] = tp/(tp+fn+sys.float_info.epsilon)
    matrix['f1'] = 2*(matrix['recall']*matrix['precision'])/(matrix['precision']+matrix['recall']+sys.float_info.epsilon)
    matrix['sensitivity'] = tp/(tp+fn+sys.float_info.epsilon)
    matrix['specificity'] = tn/(tn+fp+sys.float_info.epsilon)
    matrix['tpv'] = tp/(tp+fp+sys.float_info.epsilon)
    matrix['npv'] = tn/(tn+fn+sys.float_info.epsilon)
    matrix['tpr'] = tp/(tp+fn+sys.float_info.epsilon)
    matrix['fpr'] = 1 - matrix['specificity']
    
    # ----- 5. calculate area under the curve (auc) under the roc curve
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    matrix['auc'] = auc(fpr,tpr)
    
    # ----- return
    return matrix


def print_evaluation_matrix(matrix):
    """
    return a string of the evaluation matrix
    """
    return ''.join(['%s:%s\t' % (key, value) for (key, value) in matrix.items()])


def to_np(x):
    """
    Return a numpy from a pytorch gpu tensor
    """
    return x.data.cpu().numpy()


def get_train_test_dataset(folds, fold_index, data, dataset_transform):
    """
    Return the train and test set (with labels and pytorch dataloader) given the current fold_index.
    
    :param folds: list of folds split
    :param fold_index: the current test fold index in the folds list
    :param data: the data object that store the imgs
    :param dataset_transform: datasettransform object 
    :param is_roi_filtered: is using the precise or complete imgs
    :param num_augments: number of augmented samples
    """
    testset_patients = folds[fold_index]
    trainset_patients = [x for i,x in enumerate(folds) if i!=fold_index]
    trainset_patients = list(itertools.chain.from_iterable(trainset_patients))


    # Define the training dataset
    trainset = Dataset.MMDataset(data, trainset_patients, 
                                  dataset_transform['train'], 
                                  datasetType='Train'
                                  )
    trainset_label = np.array([data[idx]['label'] for idx in trainset_patients])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=1)

    # Define the test dataset
    testset = Dataset.MMDataset(data, testset_patients, 
                                 dataset_transform['test'],
                                 datasetType='Test'
                                 )
    testset_label = np.array([data[idx]['label'] for idx in testset_patients])
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                          shuffle=False, num_workers=1)
    
    datasets = {'train':trainset, 'test':testset}
    labels = {'train':trainset_label, 'test':testset_label}
    dataloaders = {'train':trainloader,'test':testloader}
    
    return datasets, labels, dataloaders


def plot_performance(plt, fig, ax, orderDict, title=None, xlabel=None, ylabel=None, 
                     markers=['o'], colors=['blue'],
                     markerSize=7, tickSize=14, labelSize=16, titleSize=18):
    """
    Plot an interative plot on a performance Dict (ordered).
    
    :params plt: the matplotlib.pyplot instance
    :params fig: the current plt figure object
    :params ax: the current fig axis
    :params orderDict: the performance dictionary, assume there are two performances for train- and test- set
    :params xlabel: figure x-axis label
    :params ylabel: figure y-axis label
    :params markers: list of available plot markers
    :params colors: list of available plot colors
    :params markerSize: marker size
    :params tickSize: axis tick size
    :params labelSize: axis label font size
    :params titleSize: title font size
    """
    # initate setup
    ax.clear()
    ax.tick_params(labelsize=tickSize)
    fig.show()

    # loop performance Dict
    count = 0 
    for key in orderDict.keys():
        ax.plot(orderDict[key],
                marker=markers[count%2],
                color=colors[count//2],
                markersize=markerSize,
               )
        count += 1
    
    # plot setup
    if title:
        plt.title(title, fontsize=titleSize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=labelSize)
    if ylabel:        
        plt.ylabel(ylabel, fontsize=labelSize)
    
    # legend
    ax.legend(orderDict.keys(), fontsize=labelSize, loc='upper right')
    
    # draw
    fig.canvas.draw()


def calculate_loss(net, loader, criterion, apply_log=True):
    """
    Calculate the loss from the skorch network based on provided dataloader and criterion

    10/27/18 Also return the probs and labels to favour calculation of auc 
    during training
    
    :params net: the skorch network
    :params loader: dataloader
    :params criterion: what pytorch criterion to be used to calculate the loss
    :params apply_log: apply torch.log() or not (probably yes)
    """    
    # ====================
    # Cacluating loss and probability
    # ====================
    cnn = net.module_
    cnn.train(False)
    running_loss = 0
    count = 0

    # 10/27/18 Need to get the probabilities as well to calculate the auc
    all_probs = []
    all_labels = []

    for i, img in enumerate(loader, 0):
        # get the inputs and labels
        inputs, labels = img[0].cuda(), img[1].cuda()

        # get the probabilities
        y_proba = net.predict_proba(inputs)
        y_proba = y_proba[:,1]
        all_probs.extend(y_proba.tolist())
        all_labels.extend(labels.tolist())

        # forward; get the loss
        outputs = cnn(inputs) 
        if apply_log:
            outputs = torch.log(outputs) # if I have softamx() as output in net, I need to do this
        loss = criterion(outputs, labels)

        # calculate running statistics
        num_samples = labels.cpu().numpy().shape[0]  
        running_loss += loss.item()*num_samples # need to multiple num_samples since it is average loss from pytorch
        count += num_samples   

    # store statistics
    running_loss = running_loss/count

    # probs and labels
    preds = {'probs': all_probs, 'labels': all_labels}
    preds = CustomDict(preds)
    
    return running_loss, preds
    
    
    
    
    
    
    
    
    