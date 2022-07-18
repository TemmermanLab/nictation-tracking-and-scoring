# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:03:12 2022


Contains functions used in training and evaluating models for scoring
nictation, evaluating features, and scoring nictation. Functions for 
calculating features and nictation metrics are in separate files.

Issues and improvements:
    
    -Evaluate models only takes training data from one video


@author: Temmerman Lab
"""

import numpy as np
import pandas as pd
import pickle
import csv
import copy
import os
import sys
import cv2
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

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



from pathlib import Path
home = str(Path.home())
sys.path.append(home + '//Dropbox//Temmerman_Lab//code//' + \
                'tracking-and-scoring-nictation//nictation_scoring')
sys.path.append(home + \
                r'\Dropbox\Temmerman_Lab\code\tracking-and-scoring-nictation')


import nictation_features as nf
import nictation_plotting as nict_plot
import tracker as trkr
import data_management_module as dmm

# USED
def evaluate_models_accuracy(vid_file, **kwargs):
    '''Trains and tests several types of machine learning algorithms based on
    the features calculated from and manual scores provided for <vid_file>
    split 75 / 25 training / testing.  The resulting accuracies of the 
    different models with different types of feature normalization are 
    returned as well as the training and inference times.'''   
    
    trials = kwargs.get('trials',1)
    
    model_types = kwargs.get('model_types',['logistic regression','decision tree',
                   'k nearest neighbors', 'linear discriminant analysis',
                   'Gaussian naive Bayes', 'support vector machine', 
                   'random forest', 'neural network'])
    
    scaling_methods = kwargs.get('scaling_methods',['none','min max',
                        'variance','Gaussian','whiten'])
    
    rand_split = kwargs.get('rand_split',False)
    
    # load manual scores
    man_scores_lst = load_manual_scores_csv(
        os.path.splitext(vid_file)[0] + \
        r'_tracking/manual_nictation_scores.csv')
    
    # load features
    df = pd.read_csv(os.path.splitext(vid_file)[0] + 
                      r'_tracking\nictation_features.csv')
    
    # add manual scores to df
    man_scores = []
    for scr_w in man_scores_lst:
        man_scores += list(scr_w)
    df.insert(2,'manual_behavior_label',man_scores)

    # remove NaN values
    df_masked = nan_inf_mask_dataframe(df)
    
    # plot the abundance of worm-frames with each behavior label
    nict_plot.bar_count(df_masked)
    
    acc = np.empty((len(scaling_methods),trials,len(model_types),2))
    times = copy.copy(acc)
    
    for sm in range(len(scaling_methods)):
        if scaling_methods[sm] != 'none':
            df_scaled, scaler = scale_data(df_masked, 
                                           method = scaling_methods[sm])
        else:
            df_scaled = copy.deepcopy(df_masked)
        
        for t in range(trials):
            x_train, x_test, y_train, y_test, wi_train, worminf_test = split(
                df_scaled, 0.75, rand_split)
            
            for mt in range(len(model_types)):
                print(model_types[mt])
                
                t0 = time.time()
                mod, train_acc, test_acc, probs, preds = learn_and_predict(
                    x_train, x_test, y_train, y_test, model_types[mt])
                train_time = time.time()-t0
                
                t0 = time.time()
                predictions = mod.predict(x_test) 
                infr_time = time.time()-t0
                
                acc[sm,t,mt,0] = train_acc; acc[sm,t,mt,1] = test_acc;
                times[sm,t,mt,0] = train_time
                times[sm,t,mt,1] = infr_time
                
    
    return acc, times
                # heatmap_acc[mt,2*sm:2*sm+2] = [train_acc,test_acc]


def combine_and_prepare_man_scores_and_features(vid_file, score_file = None):
    '''Loads manual scores and features from the same video and combines and
    nan-masks them in preparation for training a classifier'''
    
    # load manual scores
    if score_file is None:
        man_scores_lst = load_manual_scores_csv(
            os.path.splitext(vid_file)[0] + \
            r'_tracking/manual_nictation_scores.csv')
    else:
        man_scores_lst = load_manual_scores_csv(score_file)
    
    # load features
    df = pd.read_csv(os.path.splitext(vid_file)[0] + 
                      r'_tracking\nictation_features.csv')
    
    # add manual scores to df
    man_scores = []
    for scr_w in man_scores_lst:
        man_scores += list(scr_w)
    df.insert(2,'manual_behavior_label',man_scores)

    # remove NaN values
    df_masked = nan_inf_mask_dataframe(df)
    
    return df_masked


def evaluate_models_accuracy_2(vid_file_train, vid_file_test, **kwargs):
    '''Same as evaluate_models_accuracy except that accuracy is also evaluated
    on a totally separate video as a test of the robustness of the classifier.
    "evaluate_models_accuracy" trains and tests several types of machine
    learning algorithms based on the features calculated from and manual 
    scores provided for <vid_file> split 75 / 25 training / testing.  The
    resulting accuracies of the different models with different types of 
    feature normalization are returned as well as the training and inference
    times.'''   
    
    trials = kwargs.get('trials',1)
    
    model_types = kwargs.get('model_types',['logistic regression','decision tree',
                   'k nearest neighbors', 'linear discriminant analysis',
                   'Gaussian naive Bayes', 'support vector machine', 
                   'random forest', 'neural network'])
    
    scaling_methods = kwargs.get('scaling_methods',['none','min max',
                        'variance','Gaussian','whiten'])
    
    rand_split = kwargs.get('rand_split',False)
    
    
    # load features and manual scores for the training and test videos
    df_train = combine_and_prepare_man_scores_and_features(vid_file_train)
    df_test = combine_and_prepare_man_scores_and_features(vid_file_test)
    
    # # plot the abundance of worm-frames with each behavior label
    # nict_plot.bar_count(df_train)
    
    acc = np.empty((len(scaling_methods),trials,len(model_types),3))
    times = copy.copy(acc)
    
    for sm in range(len(scaling_methods)):
        if scaling_methods[sm] != 'none':
            
            df_train_scaled, scaler = scale_data(df_train, 
                                           method = scaling_methods[sm])
            
            cols = df_test.columns[3:]
            df_test_scaled = copy.deepcopy(df_test)
            df_test_scaled[cols] = scaler.transform(df_test[cols])
            
        else:
            df_train_scaled = copy.deepcopy(df_train)
            df_test_scaled = copy.deepcopy(df_test)
        
        for t in range(trials):
            x_train, x_test, y_train, y_test, wi_train, worminf_test = split(
                df_train_scaled, 0.75, rand_split)
            
            for mt in range(len(model_types)):
                print(model_types[mt])
                
                # train on training video and record training and test 
                # accuracy on this video and elapsed time
                t0 = time.time()
                mod, train_acc, train_test_acc, probs, preds = learn_and_predict(
                    x_train, x_test, y_train, y_test, model_types[mt])
                train_time = time.time()-t0
                
                # use the same model to make inferences on another video and
                # record the elapsed time and accuracy
                t0 = time.time()
                preds = mod.predict(
                    df_test_scaled[df_test_scaled.columns[3:]])
                test_test_acc = np.sum(np.array(df_test_scaled[
                    'manual_behavior_label']) == preds) / len(preds)
                infr_time = time.time()-t0
                
                
                acc[sm,t,mt,0] = train_acc
                acc[sm,t,mt,1] = train_test_acc
                acc[sm,t,mt,2] = test_test_acc
                
                times[sm,t,mt,0] = train_time
                times[sm,t,mt,1] = infr_time
                
    
    return acc, times
                # heatmap_acc[mt,2*sm:2*sm+2] = [train_acc,test_acc]

    # # initialize heatmap for accuracy
    # heatmap_acc = np.empty((len(model_types),10))
    # # accuracy heat map figure
    # fig, axes = plt.subplots()
    # im = axes.imshow(heatmap_acc,cmap='viridis', vmin = 0.67, vmax = 1.00)
    # #im = axes.imshow(heatmap_acc,cmap='viridis', vmin = 0.00, vmax = 1.00)
    # plt.title('Model Performance with Scaled Features')
    # axes.xaxis.set_label_position('top')
    # axes.xaxis.tick_top() 
    # axes.set_xticks([0,1,2,3,4,5,6,7,8,9])
    # axes.set_xticklabels(['train','test','train','test','train','test',
    #                       'train','test','train','test'])
    # axes.set_yticks(np.arange(len(model_types)))
    # axes.set_yticklabels(model_types)
    # axes.set_xlabel(
    #     '   none      min-max     variance   Gaussian   whitening ')
    # plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center', 
    #          rotation_mode = 'anchor')
    
    # for i in range(10):
    #     for j in range(len(model_types)):
    #         text = axes.text(i,j,"%0.2f" % heatmap_acc[j,i],ha='center',
    #                          va='center',fontweight = 'bold')
    
    # plt.savefig(os.path.splitext(vid_file)[0] + \
    #             r'_tracking/nict_scoring_model_accuracy.png', dpi = 200)
    
    # plt.show()



def separate_list_by_worm(lst, df):
    '''Takes a continuous list <lst> and splits it into a list of lists 
    based on the worm numbers in <df>'''
    
    lst = np.array(lst)
    lst_by_worm = []
    df_zeroi = df.reset_index()
    
    num_w =  int(df.loc[df['worm'].idxmax()][0]) + 1
    for w in range(num_w):
        inds = df_zeroi.index[df_zeroi['worm'] == w].tolist()
        lst_by_worm.append(lst[inds])
    
    return lst_by_worm


def compare_scores_list(man_scores, comp_scores):
    '''Takes two lists of lists of worm track scores, compares them, and
    returns the overall accuracy.'''
    
    total = 0
    same = 0
    for wt in range(len(man_scores)):
        total += len(comp_scores[wt])
        comp_scores[wt] = comp_scores[wt].astype(np.int16)
        man_scores[wt] = man_scores[wt].astype(np.int16)
        same += np.sum(comp_scores[wt]==man_scores[wt])
    return same / total
    


# from nictation_20220523\nictation_scoring_training\nict_scoring_functions
# USED
def load_manual_scores_csv(csv_file):
    '''returns the manual nictation scored in <csv_file> as a list of arrays,
    one array per worm'''
    scores_arr = []
    blank = np.nan
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
                scores_arr = score_row
            elif rc > 1:
                scores_arr = np.vstack((scores_arr,score_row))  
            rc = rc + 1
    
    scores_arr = np.rot90(scores_arr)
    scores_lst = []
    for w in reversed(range(len(scores_arr))):
        scores_lst.append(scores_arr[w][np.where(~np.isnan(scores_arr[w]))])
    
    return scores_lst



# def load_scores_csv(csv_path):
#     '''Loads the manual or automatic scores <csv_path> as a list of numpy 
#     arrays of int8 where each item is scores from one worm-track'''
#     scores = []
#     with open(csv_path) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         first_row = True
#         for row in csv_reader:
#             if first_row:
#                 for w in row:
#                     scores.append([])
#                 first_row = False
#             else:
#                 for w in range(len(row)):
#                     if row[w] != '':
#                         scores[w].append(int(row[w]))
        
#         for w in range(len(scores)):
#             scores[w] = np.array(scores[w],dtype = np.int8)
                        
#     return scores  

# USED
def nan_inf_mask_dataframe(dataframe):                               
    rows_masked_nan = 0
    rows_masked_inf = 0
    to_mask = []
    for column in dataframe: 
        #print('NaN masking ' + column)
        for i in range(len(dataframe[column])): 
            if type(dataframe[column][i]) is not str and np.isnan(dataframe[column][i]) and i not in to_mask:
                to_mask.append(i)
                rows_masked_nan += 1
            
            if type(dataframe[column][i]) is not str and np.isinf(dataframe[column][i]) and i not in to_mask:
                to_mask.append(i)
                rows_masked_inf += 1
    
    #print(str(rows_masked_nan) + ' rows masked due to NaN.')
    #print(str(rows_masked_inf) + ' rows masked due to inf.') # could be zero if NaN in same row found first (also vice versa above)
    
    df_masked = copy.deepcopy(dataframe)
    df_masked.drop(to_mask,axis=0,inplace=True)
    
    return df_masked


# USED
def scale_data(dataframe,method = 'min max'):
    
    df = copy.deepcopy(dataframe)
    
    if method == 'min max':
        scaler = MinMaxScaler()
    elif method == 'variance':
        scaler = StandardScaler()
    elif method == 'Gaussian':
        scaler = PowerTransformer(method = 'yeo-johnson')
    elif method == 'whiten':
        scaler = PCA(whiten = True)
    
    
    df_scaled = copy.deepcopy(dataframe)
    cols = dataframe.columns[3:]
    scaler = scaler.fit(df_scaled[cols])
    df_scaled[cols] = scaler.transform(df_scaled[cols])
    
    return df_scaled, scaler


# USED
def split(df_masked, prop_train = 0.75, rand_split = False):
    '''Splits the data into training and test X (features), y (manual scores),
    and wi (worm number and frame) at <prop_train>.  The index at which the
    data are split is shifted by a random amount if <rand_split>.'''
    
    X = df_masked[df_masked.columns[3:]]
    y = df_masked['manual_behavior_label']
    wi = df_masked[df_masked.columns[0:2]]
    
    if rand_split:
        # start the split at s and wrap instead of starting at zero
        offset = int(np.round(np.random.rand()*len(y)))
    else:
        offset = 0
        
    spl_ind_1 = int(np.round((prop_train)*len(y))) + offset
    spl_ind_2 = len(y) + offset
    
    if spl_ind_1 < len(y):
        X_train_spl = X[spl_ind_2%len(y):spl_ind_1]
        X_test_spl = pd.concat([X[spl_ind_1:] , X[:spl_ind_2%len(y)]])
        y_train_spl = y[spl_ind_2%len(y):spl_ind_1]
        y_test_spl = pd.concat([y[spl_ind_1:] , y[:spl_ind_2%len(y)]])
        wi_train_spl = wi[spl_ind_2%len(y):spl_ind_1]
        wi_test_spl = pd.concat([wi[spl_ind_1:] , wi[:spl_ind_2%len(y)]])
        
    else:
        X_train_spl = pd.concat([X[:spl_ind_1%len(y)] , X[spl_ind_2%len(y):]])
        X_test_spl = X[spl_ind_1%len(y):spl_ind_2%len(y)]
        y_train_spl = pd.concat([y[:spl_ind_1%len(y)] , y[spl_ind_2%len(y):]])
        y_test_spl = y[spl_ind_1%len(y):spl_ind_2%len(y)]
        wi_train_spl =  pd.concat([wi[:spl_ind_1%len(y)] , wi[spl_ind_2%len(y):]])
        wi_test_spl = wi[spl_ind_1%len(y):spl_ind_2%len(y)]
    
    return X_train_spl, X_test_spl, y_train_spl, y_test_spl, wi_train_spl, \
        wi_test_spl


# USED
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
        #print('WARNING: SVM probabilities may not correspond to scores.')
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



def load_centroids_csv(centroids_file):
    '''Reads the centroids and first frames in the .csv <centroids_file
    and returns them in the format used within the tracking code'''
    
    # load row by row
    xs = []
    ys = []
    first_frames = []
    with open(centroids_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                #print(f'Column names are {", ".join(row)}')
                row_count += 1
            elif np.mod(row_count,2)==0:
                ys.append(np.array(row[1:],dtype='float32'))
                row_count += 1
            else:
                first_frames.append(row.pop(0))
                xs.append(np.array(row,dtype='float32')); row_count += 1
    
    # reshape into the proper format
    centroids = []
    for w in range(len(xs)):
        centroids_w = []
        for f in range(len(xs[w])):
            centroids_w.append(np.array((xs[w][f],ys[w][f])))
        centroids.append(centroids_w)
    
    first_frames = [int(ff) for ff in first_frames]
    
    return centroids, first_frames



def train_model(manual_score_file,feature_file,fps,scaling_method,model_type):
    
    # load training data
    # manual_score_file = data_dir + 'Ce_R2_d21_tracking//manual_nictation_scores.csv'
    # man_scores_lst = load_manual_scores_csv(manual_score_file)
        
    # load features
    feature_file = data_dir + 'Ce_R2_d21_tracking//nictation_features.csv'
    df = pd.read_csv(feature_file)
    
    # specify frame rate
    fps = 5
    
    # add manual scores to df
    man_scores = []
    for scr_w in man_scores_lst:
        man_scores += list(scr_w)
    df.insert(2,'manual_behavior_label',man_scores)
    
    # remove NaN values
    df_masked = nan_inf_mask_dataframe(df)
    
    # specifiy machine learning parameters 
    scaling_method = 'Gaussian'
    model_type = 'random forest'
    # model_type = 'k nearest neighbors'
    
    # scale data
    df_scaled, scaler = scale_data(df_masked, method = scaling_method)
    
    
    # split up training data, use 90% of it for training this time
    x_train, x_test, y_train, y_test, worminf_train, worminf_test = split(
        df_scaled, 0.75)
    
    # train model
    mod, train_acc, test_acc, probs, preds = learn_and_predict(x_train, x_test,
                                                               y_train, y_test,
                                                               model_type)

    
    # save model
    with open('trained_'+model_type+'.pkl', 'wb') as f:
        pickle.dump([mod, train_acc, test_acc, probs, preds, model_type,
                     scaler], f)
        # with open('trained_'+model_type+'.pkl', 'rb') as f: 
        #     mod, train_acc, test_acc, probs, preds,model_type,scaler = pickle.load(f)

    return mod, train_acc, test_acc, probs, preds


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
    
    if sigma != 0:
        if type(probabilities)==list:
            for w in range(len(probabilities_smooth)):
                if sigma != 0:
                    probabilities_smooth[w] = gaussian_filter1d(
                        probabilities_smooth[w],sigma, axis = 0)
        else:
            for behavior in range(np.shape(probabilities_smooth)[1]):
                probabilities_smooth[:,behavior] = gaussian_filter1d(
                probabilities_smooth[:,behavior],sigma)

    return probabilities_smooth



def probabilities_to_predictions(probs):
    preds = list()
    
    for w in probs: preds.append(np.argmax(w,axis = -1)-1)

    return preds#np.int8(preds)

    

# def first_derivative_df(df,fps):
#     '''Calculates the first derivative of each feature based on the value in
#     the previous worm-frame and time elapsed (1/fps).  Inserts NaN if there is
#     no previous frame available'''
    
#     df_primed = copy.deepcopy(df)
    
#     for col in df.columns[2:]:
#         new_col = []
#         for row in range(np.shape(df)[0]):
#             if df['frame'][row] == 0 or row == 0 or np.isnan(df[col][row]) \
#                 or np.isnan(df[col][row-1]):
#                 new_col.append(np.nan)
#             else:
#                 new_col.append((df[col][row]-df[col][row-1])/(1.0/fps))
#         df_primed[col+'_primed'] = new_col

#     return df_primed #, df_truncated
    

def calculate_metafeatures(df,fps):
    '''Calculates several metafeatures from the original features, including
    the first derivative of each feature based on the value in the previous
    worm-frame and time elapsed (1/fps), the first derivative based on the
    value in the next worm-frame, and statistical features like the min, max,
    variance, mean, median, and total change of each feature.  Inserts NaN if
    there is no previous or or next frame value available for the derivatives,
    or if any of the neighborhood frames are missing (+/- 2 frames) for the
    statistical features.'''
    
    df_meta = copy.deepcopy(df)
    feats = copy.copy(df.columns[2:])

    
    # first derivative based on previous frame
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            # assign NaN for first frame in wormtrack, first entry in df, or
            # if the current or previous frame is nan
            if df['frame'][row] == 0 or row == 0 or np.isnan(df[col][row]) \
                or np.isnan(df[col][row-1]):
                new_col.append(np.nan)
            else:
                new_col.append((df[col][row]-df[col][row-1])/(1.0/fps))
        df_meta[col+'_primed1'] = new_col


    # first derivative based on next frame
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            # assign NaN for last frame in wormtrack, last entry in df, 
            # or if the current or nexr frame is NaN
            if row == np.shape(df)[0]-1 or df['frame'][row+1] == 0 or np.isnan(df[col][row]) \
                or np.isnan(df[col][row+1]):
                new_col.append(np.nan)
            else:
                new_col.append((df[col][row+1]-df[col][row])/(1.0/fps))
        df_meta[col+'_primed2'] = new_col


    # min
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            # assign NaN for first or last two frames in wormtrack, first or
            # last two entries in the dataframe, or if the range of +/- two
            # frames contains a nan (this is taken care of by using np.mean
            # instead of np.nanmean, etc.)
            if row <= 2 or row >= np.shape(df)[0]-3 or \
                df['frame'][row+2] <= 3:
                new_col.append(np.nan)
            else:
                new_col.append(np.min(df[col][row-2:row+3]))
        df_meta[col+'_min'] = new_col
    
    # max
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            if row <= 2 or row >= np.shape(df)[0]-3 or \
                df['frame'][row+2] <= 3:
                new_col.append(np.nan)
            else:
                new_col.append(np.max(df[col][row-2:row+3]))
        df_meta[col+'_max'] = new_col
    
    # var
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            if row <= 2 or row >= np.shape(df)[0]-3 or \
                df['frame'][row+2] <= 3:
                new_col.append(np.nan)
            else:
                new_col.append(np.var(df[col][row-2:row+3]))
        df_meta[col+'_var'] = new_col
    
    # mean
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            if row <= 2 or row >= np.shape(df)[0]-3 or \
                df['frame'][row+2] <= 3:
                new_col.append(np.nan)
            else:
                new_col.append(np.mean(df[col][row-2:row+3]))
        df_meta[col+'_mean'] = new_col
    
    # median
    for col in feats:
        new_col = []
        for row in range(np.shape(df)[0]):
            if row <= 2 or row >= np.shape(df)[0]-3 or \
                df['frame'][row+2] <= 3:
                new_col.append(np.nan)
            else:
                new_col.append(np.median(df[col][row-2:row+3]))
        df_meta[col+'_med'] = new_col

    return df_meta.copy() # .copy() de-fragments the dataframe



# def split_man_scores(man_scores, wi):
#     '''Returns a list of arrays of manual scores based on worm information
#     <wi>'''
    
#     man_scores = np.array(man_scores,dtype = object)
#     man_scores_subset = []
#     worms = np.unique(np.array(wi['worm'])).astype(np.int32)
#     frames = np.array(wi['frame']).astype(np.int32)
    
#     for w in worms:
#         man_scores_subset.append(
#             man_scores[w][frames[np.where(np.array(wi['worm'])==w)]])
    
#     return man_scores_subset



import numpy as np
import pandas as pd
import csv
import copy
from sklearn.model_selection import train_test_split

testing = False


# PREPROCESSING

def load_manual_scores_csv(csv_file):
    '''returns the manual nictation scored in <csv_file> as a list of arrays,
    one array per worm'''
    scores_arr = []
    blank = np.nan
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
                scores_arr = score_row
            elif rc > 1:
                scores_arr = np.vstack((scores_arr,score_row))  
            rc = rc + 1
    
    scores_arr = np.rot90(scores_arr)
    scores_lst = []
    for w in reversed(range(len(scores_arr))):
        scores_lst.append(scores_arr[w][np.where(~np.isnan(scores_arr[w]))])
    
    return scores_lst





# returns a dataframe with       
def nan_inf_mask_dataframe(dataframe):                               
    rows_masked_nan = 0
    rows_masked_inf = 0
    to_mask = []
    for column in dataframe: #print(column)
        for i in range(len(dataframe[column])): 
            if type(dataframe[column][i]) is not str and np.isnan(dataframe[column][i]) and i not in to_mask:
                to_mask.append(i)
                rows_masked_nan += 1
            
            if type(dataframe[column][i]) is not str and np.isinf(dataframe[column][i]) and i not in to_mask:
                to_mask.append(i)
                rows_masked_inf += 1
    
    print(str(rows_masked_nan) + ' rows masked due to NaN.')
    print(str(rows_masked_inf) + ' rows masked due to inf.') # could be zero if NaN in same row found first (also vice versa above)
    
    df_masked = copy.deepcopy(dataframe)
    df_masked.drop(to_mask,axis=0,inplace=True)
    
    return df_masked
    


def scale_data(dataframe, scaler = None ,method = 'min max'):
    
    if scaler is None:
        if method == 'min max':
            scaler = MinMaxScaler()
        elif method == 'variance':
            scaler = StandardScaler()
        elif method == 'Gaussian':
            scaler = PowerTransformer(method = 'yeo-johnson')
        elif method == 'whiten':
            scaler = PCA(whiten = True)
    
    df_scaled = copy.deepcopy(dataframe)
    cols = dataframe.columns[3:]
    df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    
    return df_scaled, scaler



def shuffle(df_masked,prop_train = 0.75):
    
    X = df_masked[df_masked.columns[2:]]
    y = df_masked['manual_behavior_label']
    
    X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = \
        train_test_split(X, y, random_state=0, test_size = 1-prop_train)
    
    return X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf



# def split_scores_by_wormtrack(scores, worm_info):
#     '''Splits the scores into a list of lists where each sublist contains the
#     behavior scores from a single wormtrack'''
    
#     scores_by_wormtrack = []
#     wi = np.array((worm_info['worm']))
#     scores = np.array(scores)
#     for w in np.unique(wi):
#         scores_by_wormtrack.append(scores[np.where(wi == w)])
        
#     return scores_by_wormtrack

# MACHINE LEARNING MODELS


def scramble_df_col(df, cols_to_scramble, rand_rand = False):
    '''Takes a dataframe <df> and randomly permutes all columns in the list
    <cols_to_scramble>. If <rand_rand> is False, then the random number
    generator is seeded for consistency. Returns the scrambled dataframe.'''
    
    if rand_rand:
        np.random.seed(0)
    
    df_scr = copy.copy(df)
    for col in df.columns:
        if col in cols_to_scramble:
            df_scr[col] = np.random.permutation(df_scr[col].values)
    
    
    return df_scr


# shuffled:
# logistic regression (can fail to converge w/ default 100 iterations)
def learn_and_predict(X_train, X_test, y_train, y_test, 
                      model_type = 'k nearest neighbors'):
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
    try:
        print('Accuracy of ',model_type,' classifier on test set: {:.2f}'
             .format(model.score(X_test, y_test)))
    except:
        import pdb; pdb.set_trace()
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return model, train_acc, test_acc, probabilities, predictions



def calculate_features(vid_file):
    
    
    gap = 1
    halfwidth = 88
    path_f = 3
    
    # load centroids, first frames, centerlines, and centerline flags
    cents, ffs  = dmm.load_centroids_csv(
        os.path.splitext(vid_file)[0] + r'_tracking\centroids.csv')
    
    clns, cln_flags = dmm.load_centerlines_csv(
        os.path.splitext(vid_file)[0] + r'_tracking\centerlines')

    
    # load tracking parameters
    params = dmm.load_parameter_csv(
        os.path.splitext(vid_file)[0] + r'_tracking\tracking_parameters.csv')
    
    
    vid = cv2.VideoCapture(vid_file)
    bkgnd = trkr.Tracker.get_background(vid,10) # needed?
    

    
    # names of features
    cols = ['worm', 'frame', 'blur', 'bkgnd_sub_blur', 
            'bkgnd_sub_blur_ends','ends_mov_bias', 'body_length',  
            'total_curvature','lateral_movement', 'longitudinal_movement', 
            'out_of_track_centerline_mov', 'angular_sweep', 'cent_path_past',
            'cent_path_fut', 'centroid_progress', 'head_tail_path_bias', 
            'centroid_path_PC_var_ratio', 'centroid_path_PC_var_product']
    
    
    # create a dataframe to hold the feature values
    df = pd.DataFrame(columns = cols)
    
    # clns = clns[0:10]; ffs = ffs[0:10]; cents = cents[0:10]
    # calculate difference image activity
    activity = nf.diff_img_act(vid, clns, ffs)
    

    # calculate other features
    scl = params['um_per_pix']
    for w in range(len(cents)):
        print('Calculating features for worm ' + str(w+1) + ' of ' + 
              str(len(cents)) + '.')
        cl0 = None
        
        for f in range(np.shape(cents[w])[0]):
            cl = clns[w][f][0]
            cent = cents[w][f]
            
                
            # features calcualted two at a time
            lat, lon = nf.lat_long_movement(cl0, cl, scl)
            rat, prod = nf.PCA_metrics(w, f, cents, path_f, scl, False)
            
            new_row = {'worm' : int(w), 'frame' : int(f), 
                        'ends_mov_bias' : nf.ends_mov_bias(cl0, cl, scl),
                        'out_of_track_centerline_mov' : 
                            nf.out_of_track_centerline_mov(cl0, cl, scl),
                        'blur' : nf.blur(vid, f+ffs[w], w, f, cent, cl, scl, 
                                        halfwidth),
                        'bkgnd_sub_blur' : nf.bkgnd_sub_blur(vid, f+ffs[w], w, 
                                        f,cent, cl, scl, halfwidth, bkgnd),
                        'bkgnd_sub_blur_ends' : 
                            nf.bkgnd_sub_ends_blur_diff(vid, f+ffs[w], w, f, 
                                                cl, scl, halfwidth, bkgnd),
                        'body_length' : nf.body_length(cl,scl),
                        'total_curvature' : nf.total_curvature(cl),
                        'lateral_movement' : lat,
                        'longitudinal_movement' : lon,
                        'angular_sweep' : nf.angular_sweep(cl0, cl, True),
                        'cent_path_past' : nf.centroid_path_length_past(w,
                                          f, cents[w], scl, path_f),
                        'cent_path_fut' : nf.centroid_path_length_fut(w,
                                          f, cents[w], scl, path_f),
                        'centroid_progress' : nf.centroid_progress(w, f, 
                                                    cents, path_f, scl),
                        'head_tail_path_bias' : nf.head_tail_path_bias(
                                                clns, w, f, path_f, scl),
                        'centroid_path_PC_var_ratio' : rat, 
                        'centroid_path_PC_var_product' : prod
                        }
            
            df = df.append(new_row,ignore_index = True)

            cl0 = copy.copy(cl)
    
    
    # tack on activity
    df.insert(2,'activity',activity)
    
    
    # # reload load indicator values
    # df = pd.read_csv(os.path.splitext(vid_file)[0] + 
    #                   r'_tracking\nictation_features.csv')
    
    
    # calculate first derivatives
    fps = vid.get(cv2.CAP_PROP_FPS)
    df = calculate_metafeatures(df,fps)
    
    
    # save indicator values
    df.to_csv(os.path.splitext(vid_file)[0] + 
              r'_tracking\nictation_features.csv', index = False)
    



# testing
if __name__ == '__main__':
    try:
        
        # vf = r"C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Celegans_vids_cropped_full\Luca_T2_Rep1_day60002 22-01-18 11-49-24_crop_1_to_300_inc_3.avi"
        # calculate_features(vf)
        
        
        vf = r"C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Ce_R2_d21.avi"
        calculate_features(vf)
        
        vf = r"C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Ce_R3_d06.avi"
        calculate_features(vf)

        
        # vid_dir = r"C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictation_dataset"
        # file_list = os.listdir(vid_dir)
        # for f in file_list[70:]:
        #     if f[-4:] == '.avi' and f[:-4]+'_tracking' in file_list:
        #         calculate_features(vid_dir + '\\' + f)
        
        
        # vf = r"C:\Users\Temmerman Lab\Desktop\test_data_for_tracking\R1d4_first_four.avi"
        # calculate_features(vf)
        
        
    except:
        
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


