# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:03:12 2022


Contains functions used in training and evaluating models for scoring
nictation, evaluating features, and scoring nictation. Functions for 
calculating features and nictation metrics are in separate files.

Issues and improvements:
    
    -Evaluate models only takes training data from one video
    -Currently features have 16 digits, they could do with fewer
    -This file is very disorganized

@author: Temmerman Lab
"""

import random
# random.seed(0)

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


# add needed module locations to path
file_name = os.path.realpath(__file__)
sys.path.append((os.path.split(file_name)[0]))
sys.path.append(os.path.split((os.path.split(file_name)[0]))[0])
    
import nictation_features as nf
import nictation_plotting as nict_plot
import nictation_metrics as nict_met
import tracker as trkr
import data_management_module as dmm


def evaluate_models_accuracy(vid_file, **kwargs):
    '''Trains and tests several types of machine learning algorithms based on
    the features calculated from and manual scores provided for <vid_file>
    split 75 / 25 training / testing.  The resulting accuracies of the 
    different models with different types of feature normalization are 
    returned as well as the training and inference times.'''   
    
    trials = kwargs.get('trials',1)
    
    model_types = kwargs.get('model_types',['logistic regression',
        'decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes', 
        'support vector machine', 'random forest', 'neural network'])
    
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
            df_scaled, scaler = scale_training_features(df_masked, method,
                                                        df_masked.columns[3:])
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


def combine_and_prepare_man_scores_and_features(vid_file, score_file = None, 
                                        simplify = True, alt_feat = False):
    '''Loads manual scores and features from the same video and combines and
    nan-masks them in preparation for training a classifier'''
    
    # load manual scores
    if score_file is None:
        man_scores_lst = load_manual_scores_csv(
            os.path.splitext(vid_file)[0] + \
            r'_tracking/manual_nictation_scores.csv', simplify)
    else:
        man_scores_lst = load_manual_scores_csv(score_file, simplify)
    
    # load features
    if not alt_feat:
        df = pd.read_csv(os.path.splitext(vid_file)[0] + 
                          r'_tracking\nictation_features.csv')
    else:
        df = pd.read_csv(os.path.splitext(vid_file)[0] + 
                          '_tracking\\' + alt_feat)
    
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
    
    model_types = kwargs.get('model_types',['logistic regression',
        'decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes',
        'support vector machine', 'random forest', 'neural network'])
    
    scaling_methods = kwargs.get('scaling_methods',['none','min max',
                        'variance','Gaussian','whiten'])
    
    rand_split = kwargs.get('rand_split',False)
    
    sigmas = kwargs.get('sigmas', np.arange(0,1.5,0.1))
    
    fps = kwargs.get('fps', 5)
    
    only_active = kwargs.get('only_active', False)
    
    
    # load features and manual scores for the training and test videos
    df_train = combine_and_prepare_man_scores_and_features(vid_file_train)
    df_test = combine_and_prepare_man_scores_and_features(vid_file_test)
    man_scores_test_vid = separate_list_by_worm(
        df_test['manual_behavior_label'], df_test)
    
    # # plot the abundance of worm-frames with each behavior label
    # nict_plot.bar_count(df_train)
    
    accs = np.empty((len(scaling_methods),trials,len(model_types),len(sigmas)
                    ,3))
    times = np.empty((len(scaling_methods),trials,len(model_types),2))
    NRs = copy.copy(accs)
    IRs = copy.copy(accs)
    SRs = copy.copy(accs)
    
    for sm in range(len(scaling_methods)):
        if scaling_methods[sm] != 'none':
            df_train_scaled, scaler = scale_training_features(df_train, 
                                    scaling_methods[sm], df_train.columns[3:])
            
            cols = df_test.columns[3:]
            df_test_scaled = copy.deepcopy(df_test)
            df_test_scaled[cols] = scaler.transform(df_test[cols])
            
        else:
            df_train_scaled = copy.deepcopy(df_train)
            df_test_scaled = copy.deepcopy(df_test)
        
        for t in range(trials):
            x_train, x_test, y_train, y_test, wi_train, wi_test = split(
                df_train_scaled, 0.75, rand_split)
            man_scores_train_train = separate_list_by_worm(y_train, wi_train)
            man_scores_train_test = separate_list_by_worm(y_test, wi_test)
            
            for mt in range(len(model_types)):
                print(model_types[mt])
                
                # train on training video and record training and test 
                # accuracy on this video and elapsed time
                t0 = time.time()
                categories = np.unique(y_train) # sometimes a label is not
                # represented in the training set
                mod, train_acc, train_test_acc, probs, preds = \
                    learn_and_predict(x_train, x_test, y_train, y_test,
                                      model_types[mt])
                train_time = time.time()-t0
                
                
                # make probability infrerence on the training and test 
                # portions of the training video to be used below
                probs_train_train = mod.predict_proba(x_train)
                probs_train_test = mod.predict_proba(x_test)
                
                # use the same model to make inferences on another video and
                # record the elapsed time and accuracy
                t0 = time.time()
                probs_test_vid = mod.predict_proba(
                    df_test_scaled[df_test_scaled.columns[3:]])
                infr_time = time.time()-t0
                
                # cycle through smoothing sigmas
                for sg in range(len(sigmas)):
                    
                    # smooth and inferences and calculate nictation metrics on
                    # the training set of the training video
                    probs_smooth = smooth_probabilities(
                        probs_train_train, sigmas[sg], fps)
                    preds_smooth = probabilities_to_predictions(probs_smooth,
                                                                categories)
                    preds_smooth_list = separate_list_by_worm(
                        preds_smooth, wi_train)
                
                    NRs[sm,t,mt,sg,0] = nict_met.nictation_ratio(
                        preds_smooth_list, only_active)
                    IRs[sm,t,mt,sg,0] = nict_met.initiation_rate(
                        preds_smooth_list, only_active)
                    SRs[sm,t,mt,sg,0] = nict_met.stopping_rate(
                        preds_smooth_list, only_active)
                    accs[sm,t,mt,sg,0] = compare_scores_list(
                        man_scores_train_train,preds_smooth_list)
                    
                    
                    # smooth and inferences and calculate nictation metrics on
                    # the test set of the training video
                    probs_smooth = smooth_probabilities(
                        probs_train_test, sigmas[sg], fps)
                    preds_smooth = probabilities_to_predictions(probs_smooth,
                                                                categories)
                    preds_smooth_list = separate_list_by_worm(
                        preds_smooth, wi_test)
                
                    NRs[sm,t,mt,sg,1] = nict_met.nictation_ratio(
                        preds_smooth_list, only_active)
                    IRs[sm,t,mt,sg,1] = nict_met.initiation_rate(
                        preds_smooth_list, only_active)
                    SRs[sm,t,mt,sg,1] = nict_met.stopping_rate(
                        preds_smooth_list, only_active)
                    accs[sm,t,mt,sg,1] = compare_scores_list(
                        man_scores_train_test, preds_smooth_list)
                    
                    
                    # smooth and inferences and calculate nictation metrics on
                    # the separate test video
                    probs_smooth = smooth_probabilities(
                        probs_test_vid, sigmas[sg], fps)
                    preds_smooth = probabilities_to_predictions(probs_smooth,
                                                                categories)
                    preds_smooth_list = separate_list_by_worm(
                        preds_smooth, df_test_scaled)
                
                    NRs[sm,t,mt,sg,2] = nict_met.nictation_ratio(
                        preds_smooth_list, only_active)
                    IRs[sm,t,mt,sg,2] = nict_met.initiation_rate(
                        preds_smooth_list, only_active)
                    SRs[sm,t,mt,sg,2] = nict_met.stopping_rate(
                        preds_smooth_list, only_active)
                    accs[sm,t,mt,sg,2] = compare_scores_list(
                        man_scores_test_vid, preds_smooth_list)
               
                
                # training and inference times do not include the smoothing
                # time, which is basically negligable
                times[sm,t,mt,0] = train_time
                times[sm,t,mt,1] = infr_time
                
    
    return accs, times, NRs, IRs, SRs
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
    



def scale_scoring_features(dataframe, scaler, columns):
    '''Scales the specified columns accorind to the scaler provided and 
    returns a dataframe with those columns scaled'''
    
    if scaler is not None:
        df_scaled = copy.deepcopy(dataframe)
        df_scaled[columns] = scaler.transform(df_scaled[columns])
    else:
        df_scaled = copy.deepcopy(dataframe)
    
    return df_scaled




def scale_training_features(dataframe, method, columns):
    '''Fits a scaler to the specified columns, uses it to scale those columns,
    and returns a dataframe containing only those columns scaled along with \
    the scaler'''
    
    if method != 'none':
        if method == 'min max':
            scaler = MinMaxScaler()
        elif method == 'variance':
            scaler = StandardScaler()
        elif method == 'Gaussian':
            scaler = PowerTransformer(method = 'yeo-johnson')
        elif method == 'whiten':
            scaler = PCA(whiten = True)
        else:
            print('Scaling method not recognized')
        
        df_scaled = copy.deepcopy(dataframe)
        scaler = scaler.fit(df_scaled[columns])
        df_scaled[columns] = scaler.transform(df_scaled[columns])
    
    else:
        scaler = None
        df_scaled = copy.deepcopy(dataframe)
    
    return df_scaled, scaler



def split(df_masked, prop_train = 0.75, rand_split = False):
    '''Splits the data into training and test X (features), y (manual scores),
    and wi (worm number and frame) at <prop_train>.  The index at which the
    data are split is shifted by a random amount if <rand_split>.'''
    
    X = df_masked[df_masked.columns[4:]]
    y = df_masked['manual_behavior_label']
    wi = df_masked[df_masked.columns[0:3]]
    
    if rand_split:
        # start the split at s and wrap instead of starting at zero
        offset = int(np.round(random.random()*len(y)))
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
        wi_train_spl =  pd.concat([wi[:spl_ind_1%len(y)] , 
                                   wi[spl_ind_2%len(y):]])
        wi_test_spl = wi[spl_ind_1%len(y):spl_ind_2%len(y)]
    
    return X_train_spl, X_test_spl, y_train_spl, y_test_spl, wi_train_spl, \
        wi_test_spl




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
    df_scaled, scaler = scale_training_data(df_masked, scaling_method,
                                            df_masked[3:])
    
    
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

    data = {'behavior label': behavior_scores,'behavior':behavior,
            'activity': activity,
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
                probabilities_smooth[w] = gaussian_filter1d(
                    probabilities_smooth[w],sigma, axis = 0)
        else:
            for behavior in range(np.shape(probabilities_smooth)[1]):
                probabilities_smooth[:,behavior] = gaussian_filter1d(
                probabilities_smooth[:,behavior],sigma)

    return probabilities_smooth



def probabilities_to_predictions(probs, categories):
    pred_nums = list()
    
    for f in probs: pred_nums.append(np.argmax(f,axis = -1))
    
    preds = list()
    
    for i in pred_nums: preds.append(categories[i])

    return preds#np.int8(preds)

    

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
    feats = copy.copy(df.columns[3:])

    
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
    df_meta = df_meta.copy() # prevents fragmentation


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
    df_meta = df_meta.copy() # prevents fragmentation


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
    df_meta = df_meta.copy() # prevents fragmentation
    
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
    df_meta = df_meta.copy() # prevents fragmentation
    
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
    df_meta = df_meta.copy() # prevents fragmentation
    
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
    df_meta = df_meta.copy() # prevents fragmentation
    
    
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




import numpy as np
import pandas as pd
import csv
import copy
from sklearn.model_selection import train_test_split

testing = False


def load_manual_scores_csv(csv_file, simplify = True):
    '''Returns the manual nictation scored in <csv_file> as a list of arrays,
    one array per worm.  If <simplify> is True, then it changes nictation
    scores from quiescent / crawling / waving / standing to recumbent / 
    nictating'''
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
    
    if simplify:
        scores_arr[np.where(scores_arr == 1)] = 0
        scores_arr[np.where(scores_arr == 2)] = 1
        scores_arr[np.where(scores_arr == 3)] = 1


    scores_arr = np.rot90(scores_arr)
    scores_lst = []
    for w in reversed(range(len(scores_arr))):
        scores_lst.append(scores_arr[w][np.where(~np.isnan(scores_arr[w]))])
    
    return scores_lst


# returns a dataframe with       
def nan_inf_mask_dataframe(dataframe):                               
    '''Removes all -np.inf, np.inf, and np.nan values from dataframe and 
    resets the indices'''
    
    df_masked = copy.deepcopy(dataframe)
    df_masked.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_masked = df_masked.dropna().reset_index(drop=True)
    
    return df_masked
    


def shuffle(df_masked,prop_train = 0.75):
    
    X = df_masked[df_masked.columns[2:]]
    y = df_masked['manual_behavior_label']
    
    X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = \
        train_test_split(X, y, random_state=0, test_size = 1-prop_train)
    
    return X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf



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




# USED
def learn_and_predict(X_train, X_test, y_train, y_test,
                      model_type = 'random forest', print_acc = True):
    
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
    
    if print_acc:
        print('Accuracy of ',model_type,' classifier on training set: {:.2f}'
             .format(model.score(X_train, y_train)))
        
        if len(X_test) > 0:
            print('Accuracy of ',model_type,' classifier on test set: {:.2f}'
                 .format(model.score(X_test, y_test)))
    
    train_acc = model.score(X_train, y_train)
    if len(X_test) > 0:
        test_acc = model.score(X_test, y_test)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
    
    if len(X_test) > 0:
        return model, train_acc, test_acc, probabilities, predictions
    else:
        return model, train_acc



def calculate_features(vid_file, tracking_method = 'mRCNN'):
    
    tracking_method = ''#'_' + tracking_method
    
    gap = 1
    halfwidth = 88
    path_f = 3
    
    
    # load centroids, first frames, centerlines, and centerline flags
    cents, ffs  = dmm.load_centroids_csv(
        os.path.splitext(vid_file)[0] + tracking_method + \
            r'_tracking\centroids.csv')
    
    clns, cln_flags = dmm.load_centerlines_csv(
        os.path.splitext(vid_file)[0] + tracking_method + \
            r'_tracking\centerlines')

    
    # load tracking parameters
    params = dmm.load_parameter_csv(
        os.path.splitext(vid_file)[0] + tracking_method \
            + r'_tracking\tracking_parameters.csv')
    
    
    vid = cv2.VideoCapture(vid_file)
    bkgnd = trkr.Tracker.get_background(vid,10) # needed?
    

    
    # names of features
    cols = ['worm', 'frame', 'video_frame', 'blur', 'bkgnd_sub_blur', 
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
            
                
            # features calculated two at a time
            lat, lon = nf.lat_long_movement(cl0, cl, scl)
            rat, prod = nf.PCA_metrics(w, f, cents, path_f, scl, False)
            
            # other information and features
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
                        'centroid_path_PC_var_product' : prod,
                        'video_frame' : int(f + ffs[w])
                        }
            
            
            df = pd.concat([df, pd.DataFrame(new_row, index = [0])], axis = 0,
                           ignore_index=True)

            cl0 = copy.copy(cl)
    
    
    # tack on activity
    df.insert(3,'activity',activity)
    
    # also include actual video frames
    
    # # reload load indicator values
    # df = pd.read_csv(os.path.splitext(vid_file)[0] + 
    #                   r'_tracking\nictation_features.csv')
    
    
    # calculate first derivatives
    fps = vid.get(cv2.CAP_PROP_FPS)
    df = calculate_metafeatures(df,fps)
    
    
    # save feature values
    df.to_csv(os.path.splitext(vid_file)[0] + tracking_method + \
              r'_tracking\nictation_features.csv', index = False)
    

def score_behavior(feature_file, behavior_model_file, behavior_sig, fps,
                   save_path, save_scores = True, return_scores = False):
    '''Applies the model and scaler in <behavior_model_file> to the features
    in <feature_file> (the model needs to have been trained on the same types
    of features), smooths the probabilities by <behavior_sig> * <fps>, and
    saves the resulting scores and the original, unsmoothed probabilities as a
    .csv in <save_path>'''
    
    # load model and scaler
    with open(behavior_model_file, 'rb') as f: 
        mod, scaler = pickle.load(f)
    
    # load features
    df = pd.read_csv(feature_file)
    df_masked = nan_inf_mask_dataframe(df)
    dfi =  df.columns.get_loc('activity')
    cols = df.columns[dfi:]
    df_scaled = scale_scoring_features(df_masked, scaler, cols)
    df_ready = df_scaled[df_scaled.columns[dfi:]]
        
    # smooth predictions
    probs = mod.predict_proba(df_ready)
    probs_smooth = smooth_probabilities(probs, behavior_sig, fps)
    categories = list(np.arange(-1,np.shape(probs)[1]-1))
    predictions = probabilities_to_predictions(probs_smooth,
                                                       categories)
    # save predictions
    df_preds_1 = copy.deepcopy(df_masked[['worm','frame','video_frame']])
    preds_dict = {} # empty dictionary
    preds_dict['pred. behavior']=list(predictions)
    for i in range(np.shape(probs)[1]):
        preds_dict['prob. '+str(categories[i])] = list(probs[:,i])
    df_preds_2 = pd.DataFrame(preds_dict)
    df_preds = pd.concat([df_preds_1, df_preds_2], axis=1)
    
    if save_scores:
        df_preds.to_csv(save_path + r'\computer_behavior_scores.csv',
                        index = False)
    if return_scores:
        return df_preds
    
    



def Junho_Lee_scores(scores, activity, fps = 5, assume_active = True):
    '''Scours the data for worm tracks that can be scored in a way similar to
    that described in Lee 2012 and used by Bram in his studies'''
    #import pdb; pdb.set_trace()
    
    
    
    # keep only tracks of at least 1 min duration
    keep = np.where(np.array([len(s) for s in scores]) >= 300)[0]
    scores = [scores[i] for i in keep]
    activity = [activity[i] for i in keep]
    
    
    # keep only tracks with no quiescence the first minute
    # (the second minute is dealt with later)
    if not assume_active:
        keep = np.where(
            np.array([np.min(np.array(a)[0:300]) for a in activity])>0)[0]
        scores = [scores[i] for i in keep]
        activity = [activity[i] for i in keep]
    
    
    # keep only tracks with no censorship the first minute
    # (the second minute is dealt with later)
    keep = np.where(
        np.array([np.min(np.array(s)[0:300]) for s in scores])>-1)[0]
    scores = [scores[i] for i in keep]
    activity = [activity[i] for i in keep]
    
    
    # keep only tracks with where the worm is not nictating at the start
    keep = np.where(np.array([s[0] == 0 for s in scores]) == 1)[0]
    scores = [scores[i] for i in keep]
    activity = [activity[i] for i in keep]
    
    
    # trim tracks that are not nictating at 1 min to 1 min, also eliminate
    # tracks with censored frames
    keep = []
    for i, s in enumerate(scores):
        a = activity[i]
        
        # truncate to 1 min tracks that are not nictating at 1 min
        if s[299] == 0:
            s = s[0:300]
            a = a[0:300]
            keep.append(i)
        
        # keep tracks longer than one minute until they stop nictating or 
        # reach 2 min, whichever is first, unless they become quiescent or are
        # censored before this, or if they end prematurely while still
        # nictating
        elif s[299] == 1:
            # truncate tracks to two min or end of nictation, whichever is
            # first
            try:
                trunc_frame = np.where(s[300:]==0)[0][0]+300
                if len(trunc_frame) == 0: trunc_frame = 600
                kill = False
            except:
                # temporarily keep tracks that end before 2 min but are still
                # nictating, but set <kill> to True
                trunc_frame = np.max((600,len(s)))
                kill = False
                if trunc_frame == len(s):
                    kill = True
                
            s = s[:trunc_frame]
            a = a[:trunc_frame]
        
            # only keep these tracks if they do not contain censorship or
            # quiescence and did not end prematurely while still nictating
            # (kill)
            if not assume_active:
                if np.min(s) > -1 and np.min(a) > 0 and not kill:
                    scores[i] = s
                    activity[i] = a
                    keep.append(i)
            else:
                if np.min(s) > -1 and not kill:
                    scores[i] = s
                    activity[i] = a
                    keep.append(i)
    
    scores = [scores[i] for i in keep]
    activity = [activity[i] for i in keep]        
    
    
    # For each track, record the total amount of time, the amount of time
    # spent nictating, and the number of nictation bouts (initiations).  Bouts
    # ongoing at 2 min are truncated.
    tot_time = np.array([len(s) / fps for s in scores])
    nict_time = np.array([sum(s == 1) / fps for s in scores])
    N = []
    for s in scores:
        n = 0
        for i, f in enumerate(s[:-1]):
            if f == 0 and s[i+1] == 1:
                n += 1
        N.append(n)
    N = np.array(N)
    
    nict_rat = nict_time / tot_time
    init_ind = N / (tot_time - nict_time)
    avg_dur = N / nict_time
    
    # calculate total nictation time, nictation ratio, initiation index, and
    # average duration for each track
    
    
    return nict_rat, init_ind, avg_dur, tot_time, nict_time, N


# testing
if __name__ == '__main__':
    try:
        
        # vf = r"C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Celegans_vids_cropped_full\Luca_T2_Rep1_day60002 22-01-18 11-49-24_crop_1_to_300_inc_3.avi"
        # calculate_features(vf)
        
        
        
        
        # vf = r"C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Ce_R2_d21.avi"
        # calculate_features(vf)
        
        # vf = r"C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Ce_R3_d06.avi"
        # calculate_features(vf)

        
        vid_dir = r"D:\Data_flp_7_updated"
        file_list = sorted(os.listdir(vid_dir))
        for f in file_list[:]:
            if f[-4:] == '.avi' and f[:-4]+'_tracking' in file_list:
                calculate_features(vid_dir + '\\' + f)
        
        # vid_dir = r"D:\Pat working\Scarpocapsae_nictation_dataset"
        # file_list = os.listdir(vid_dir)
        # for f in file_list[0:]:
        #     if f[-4:] == '.avi' and f[:-4]+'_mRCNN_tracking' in file_list:
        #         calculate_features(vid_dir + '\\' + f)
                
        # vid_dir = r"D:\Data_flp-7_downsampled_5min"
        # file_list = os.listdir(vid_dir)
        # for f in file_list[0:]:
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


