# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:42:03 2021


This file contains the following types of functions:


1. functions for pre-processing the indicators (scaling, whitening, etc)

2. wrappers for a number of different simple machine-learning algorithms for
    classifying dauer behavior based on custom indicators of nictation
    behavior

3. functions for post-processing the indicator returns (smoothing, etc)

4. functions for evaluating the performance of models based on quantitative
    nictation metrics


Many of the metrics closely follow this example:
https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2


@author: PDMcClanahan
"""

import numpy as np
import pandas as pd
import csv
import copy
from sklearn.model_selection import train_test_split

testing = False


# PREPROCESSING

def load_manual_scores_csv(csv_file):
    scores = []
    blank = -2
    rc = 0
    
    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        for row in csv_reader:
            score_row = np.empty(len(row))
            for w in range(len(score_row)):
                if rc > 0:
                    
                    if len(row[w]) == 0:
                        score_row[w] = blank
                    else:
                        score_row[w] = int(row[w])
                    
            if rc == 1:
                scores = score_row
            elif rc > 1:
                scores = np.vstack((scores,score_row))  
            rc = rc + 1
    scores = np.rot90(scores)    
    
    
    return scores


def nan_mask_dataframe_old(dataframe):
    
    
    
    
    data = dataframe.to_dict()
    data_masked = copy.deepcopy(data)
    to_mask = []
    #import pdb; pdb.set_trace()
    for key in data_masked: #print(key)
        for i in range(len(data_masked[key])): 
            if type(data_masked[key][i]) is not str and np.isnan(data_masked[key][i]) and i not in to_mask:
                to_mask.append(i)
            # these lines flag censored frames for removal
            # if type(data_masked[key][i]) is str and data_masked[key][i] == 'censored' and i not in to_mask:
            #     to_mask.append(i)
                
    to_mask.sort(reverse=True)
    
    for key in data_masked:
        for i in to_mask:
            data_masked[key].pop(i)
    
    df_masked = pd.DataFrame(data=data_masked)
    
    return df_masked

# def nan_mask_dataframe(dataframe):
#     df_masked = copy.deepcopy(dataframe)
    
#     for column in df_masked:
#         df_masked.drop(df_masked.index[(np.isnan(df_masked[column]))],axis=0,inplace=True)
#     return df_masked


# returns a dataframe with       
def nan_mask_dataframe(dataframe):                               
    to_mask = []
    for column in dataframe: #print(column)
        for i in range(len(dataframe[column])): 
            if type(dataframe[column][i]) is not str and np.isnan(dataframe[column][i]) and i not in to_mask:
                to_mask.append(i)
    
    df_masked = copy.deepcopy(dataframe)
    df_masked.drop(to_mask,axis=0,inplace=True)
    
    return df_masked
    
    # data = dataframe.to_dict()
    # data_masked = copy.deepcopy(data)
    # to_mask = []
    # #import pdb; pdb.set_trace()
    # for key in data_masked: #print(key)
    #     for i in range(len(data_masked[key])): 
    #         if type(data_masked[key][i]) is not str and np.isnan(data_masked[key][i]) and i not in to_mask:
    #             to_mask.append(i)
    #         # these lines flag censored frames for removal
    #         # if type(data_masked[key][i]) is str and data_masked[key][i] == 'censored' and i not in to_mask:
    #         #     to_mask.append(i)
                
    # to_mask.sort(reverse=True)
    
    # for key in data_masked:
    #     for i in to_mask:
    #         data_masked[key].pop(i)
    
    # df_masked = pd.DataFrame(data=data_masked)
    
    # return df_masked



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

def scale_data(dataframe,method = 'min max'):
    
    if method == 'min max':
        scaler = MinMaxScaler()
    elif method == 'variance':
        scaler = StandardScaler()
    elif method == 'Gaussian':
        scaler = PowerTransformer(method = 'yeo-johnson')
    elif method == 'whiten':
        scaler = PCA(whiten = True)
        
    df_scaled = copy.deepcopy(dataframe)
    cols = dataframe.columns[2:]
    df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    
    return df_scaled, scaler


def shuffle(df_masked,prop_train = 0.75):
    
    X = df_masked[df_masked.columns[2:]]
    y = df_masked['behavior label']
    
    X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = \
        train_test_split(X, y, random_state=0, test_size = 1-prop_train)
    
    return X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf


def split(df_masked,prop_train = 0.75):
    
    X = df_masked[df_masked.columns[2:]]
    y = df_masked['behavior label']
    spl_ind = int(np.round(prop_train*len(y)))
    
    X_train_spl = X[0:spl_ind]
    X_test_spl = X[spl_ind:]
    y_train_spl = y[0:spl_ind]
    y_test_spl = y[spl_ind:]
    
    return X_train_spl, X_test_spl, y_train_spl, y_test_spl



# MACHINE LEARNING MODELS

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# shuffled:
# logistic regression (can fail to converge w/ default 100 iterations)
def learn_and_predict(X_train, X_test, y_train, y_test, model_type = 'k nearest neighbors'):
    if model_type == 'logistic regression':
        model = LogisticRegression(max_iter = 1000)
    elif model_type == 'decision tree':
        model = DecisionTreeClassifier()
    elif model_type == 'k nearest neighbors':
        model = KNeighborsClassifier()
    elif model_type == 'linear discriminant analysis':
        model = LinearDiscriminantAnalysis()
    elif model_type == 'Gaussian naive Bayes':
        model = GaussianNB()
    elif model_type == 'support vector machine':
        model = SVC(probability = True)
        print('WARNING: SVM probabilities may not correspond to scores.')
    elif model_type == 'random forest':
        model = RandomForestClassifier(max_features='sqrt')
    elif model_type == 'neural network':
        model = MLPClassifier()
    else:
        print('WARNING: model type "'+model_type+'" not recognized!')
    model.fit(X_train, y_train)
    print('Accuracy of ',model_type,' classifier on training set: {:.2f}'
         .format(model.score(X_train, y_train)))
    print('Accuracy of ',model_type,' classifier on test set: {:.2f}'
         .format(model.score(X_test, y_test)))
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return model, train_acc, test_acc, probabilities, predictions



# POST PROCESSING

def worm_to_df(w,metric_labels,behavior_scores,activity,metric_scores):
    
    #print('WARNING: this function will have to be changed once metric lengths are regularized')
    
    if w >= 6:
        activity = activity[1:]
        #print('Adjusting length of activity for late worm')
    
    behavior = list(copy.copy(behavior_scores))
    for i in range(len(behavior)):
        if behavior_scores[i] == 0: behavior[i] = 'quiescent'
        if behavior_scores[i] == 1: behavior[i] = 'cruising'
        if behavior_scores[i] == 2: behavior[i] = 'waving'
        if behavior_scores[i] == 3: behavior[i] = 'standing'
        if behavior_scores[i] == -1: behavior[i] = 'censored'

    data = {'behavior label': behavior_scores,'behavior':behavior, 'activity': activity,
        metric_labels[1]:metric_scores[1][w][1:],
        metric_labels[2]:metric_scores[2][w][1:],
        metric_labels[3]:metric_scores[3][w][1:],
        metric_labels[4]:metric_scores[4][w][1:],
        metric_labels[5]:metric_scores[5][w][1:],
        metric_labels[6]:metric_scores[6][w][1:],
        metric_labels[7]:metric_scores[7][w][1:],
        metric_labels[8]:metric_scores[8][w][1:],
        metric_labels[9]:metric_scores[9][w][1:],
        }

    df = pd.DataFrame(data=data)
        
    return df


from scipy.ndimage import gaussian_filter1d
def smooth_probabilities(probabilities, sigma, fps = 5.0):
    sigma = sigma * fps
    probabilities_smooth = copy.deepcopy(probabilities)
    
    if type(probabilities)==list:
        for w in range(len(probabilities_smooth)):
            if sigma != 0:
                probabilities_smooth[w] = gaussian_filter1d(probabilities_smooth[w],sigma, axis = 0)
    else:
        probabilities_smooth = gaussian_filter1d(probabilities_smooth,sigma)
            
    return probabilities_smooth



def probabilities_to_predictions(probs):
    preds = list()
    
    for w in probs: preds.append(np.argmax(w,axis = -1)-1)

    return preds#np.int8(preds)


def first_derivative_act(act,fps):
    # uint to float to handle negatives and non-integers
    for w in range(len(act)): act[w] = list(map(float, act[w]))

    act_primed = copy.deepcopy(act)
    act_trunc = copy.deepcopy(act)
    for w in range(len(act)):
        act_primed[w] = np.diff(act[w])/(1.0/fps)
        act_trunc[w] = np.array(act[w][1:])
    return act_primed, act_trunc

def first_derivative_met(met,fps):
    met_primed = copy.deepcopy(met)
    met_trunc = copy.deepcopy(met)
    for m in range(len(met)):
        for w in range(len(met[m])):
            met_primed[m][w] = np.diff(met[m][w])/(1.0/fps)
            met_trunc[m][w] = met[m][w][1:]
    return met_primed, met_trunc
    
    

def first_derivative_df(df,fps):
    df_primed = copy.deepcopy(df)
    for row in range(1,np.shape(df)[0]):
        for col in df.columns[2:]:
            df_primed.at[row, col] = (df[col][row]-df[col][row-1])/(1.0/fps)
    df_primed = df_primed.drop(df.index[0])
    df_truncated = df.drop(df.index[0])
    return df_primed, df_truncated
    

# unfinished
def predict_from_model(model,scores):    
    
    predictions = model.predict(scores)
    
    return predictions






