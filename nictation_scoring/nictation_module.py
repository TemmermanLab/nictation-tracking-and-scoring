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
    -the split function and possibly others imply incorrect terminology (test
     should be val)

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

# sometimes there are convergence warnings for some models, e.g. logistic
# regression. Rather than wait longer, I suppress the warnings.
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# add needed module locations to path
file_name = os.path.realpath(__file__)
sys.path.append((os.path.split(file_name)[0]))
sys.path.append(os.path.split((os.path.split(file_name)[0]))[0])
    
import nictation_features as nf
import nictation_plotting as nict_plot
import nictation_metrics as nict_met
import tracker as trkr
import data_management_module as dmm



# def load_and_clean_training_data(manual_score_file, feature_file):
#     '''Combines the manual scores in <manual_score_file> with the features in
#     <feature file>, removes censored worm-frames, worm-frames containing nan
#     or inf, and flagged centerlines'''
    
    
    
#     return blah

# deprecated
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
    df_masked, ix = nan_inf_mask_dataframe(df)
    
    # remove unfixed centerlines
    
    
    return df_masked, ix


def remove_censored_frames(df):
    '''Removes from the dataframe rows in which the frame was scored as
    censored, frames one or two frames away, provided they are part of the 
    same worm track. In the future, it could also remove disconnected parts of
    tracks that are left behind'''
    
    df = df.reset_index()
   
    i_del = []
    for f in range(len(df)):
        if df['manual_behavior_label'][f] == -1:
            i_del.append(f)
            
            w = df['worm'][f]
            for offset in [-2,-1,1,2]:
                if f+offset >=0 and f+offset < len(df):
                    if df['worm'][f+offset] == w and f+offset not in i_del:
                        i_del.append(f+offset)
    
    df = df.drop(index=i_del)
    # df = df.reset_index()
    # df = df.drop(columns = 'level_0')
    return df.reset_index().drop(columns = ['index','level_0'])



def clean_dataset(df):
    '''Combines removal of worm-frames containing features with NaN or inf
    values, the removal of censored and censored-adjacent frames, and the
    removal of worm-frames with flagged centerlines (and those adjacent). Here
    "adjacent" means two frames before or after. This is to avoid problems
    with statistical features calculated over several frames.'''
    
    df = df.reset_index()
    
    
    # find indices of worm-frames with NaN or inf in the features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    i_del_naninf = list(df[df.isna().any(axis=1)].index)
        
    
    # find indices of censored and censored-adjacent worm-frames within the
    # same track
    i_del_cen = []
    for f in range(len(df)):
        if df['manual_behavior_label'][f] == -1:
            i_del_cen.append(f)
            w = df['worm'][f]
            for offset in [-2,-1,1,2]:
                if f+offset >=0 and f+offset < len(df):
                    if df['worm'][f+offset] == w and f+offset not in i_del_cen:
                        i_del_cen.append(f+offset)
    
    
    # find indices of centerline-flagged worm-frames and those adject within
    # the same track
    ix_prime = list(np.where(df['centerline_flag']==1)[0])+\
        list(np.where(df['centerline_flag']==3)[0])
    ix_adj = []
    for i in ix_prime:
        for offset in [-2,-1,1,2]:
            if i+offset > 0 and i+offset < len(df) and \
                df.iloc[i]['worm'] == df.iloc[i+offset]['worm']:
                ix_adj.append(i+offset)
    i_del_flag = list(np.unique(np.array(ix_prime+ix_adj)))
    
    
    # remove the problem worm-frames
    i_del = list(np.sort(np.unique(np.array(
        i_del_naninf+i_del_cen+i_del_flag))))
    df = df.drop(index=i_del)
    
    
    return df.reset_index().drop(columns = ['index','level_0'])
    
    
def train_behavior_classifier(train_data_dir, scaling_method, algorithm):
    '''Uses the manual scores and features in <train_data_dir>, scales the
    features by <scaling_method>, and trains a classifier of type <algorithm>
    on these training data, returning the trained model / classifier and 
    scaler''' 
    # for testing
    # train_data_dir = r'C:\Users\PDMcClanahan\Desktop\test_behavior_training_set'
    # scaling_method = 'whiten'
    # algorithm = 'random forest'
    
    
    # load features
    feature_file = train_data_dir + '//nictation_features.csv'
    df = pd.read_csv(feature_file)
    
    
    # load manual scores    
    manual_score_file =  train_data_dir + '//manual_nictation_scores.csv'
    man_scores_lst = load_manual_scores_csv(manual_score_file, 
                                            simplify = False)
    man_scores = []
    for scr_w in man_scores_lst:
        man_scores += list(scr_w)
    df.insert(2,'manual_behavior_label',man_scores)
    
    
    # remove NaN values
    df_masked, ix = nan_inf_mask_dataframe(df)
    

    # scale data
    df_scaled, scaler = scale_training_features(df_masked, scaling_method,
                                            df_masked.columns[5:])
    
    
    model = fit_model(df_scaled[df_scaled.columns[5:]], \
              df_scaled['manual_behavior_label'], algorithm)
    
    return model, scaler


def fit_model(x,y,algorithm):
    # initialize model, note GNB has no random state option
    if algorithm == 'logistic regression':
        model = LogisticRegression(max_iter = 1000, random_state = 0)
    elif algorithm == 'decision tree':
        model = DecisionTreeClassifier(random_state = 0)
    elif algorithm == 'k nearest neighbors':
        model = KNeighborsClassifier()
    elif algorithm == 'linear discriminant analysis':
        model = LinearDiscriminantAnalysis()
    elif algorithm == 'Gaussian naive Bayes':
        model = GaussianNB()
    elif algorithm == 'support vector machine':
        model = SVC(probability = True,random_state = 0)
    elif algorithm == 'random forest':
        model = RandomForestClassifier(max_features='sqrt', random_state = 0)
    elif algorithm == 'neural network':
        model = MLPClassifier(random_state = 0)
    else:
        print('WARNING: algorithm type "'+algorithm+'" not recognized!')
    
    model.fit(x, y)
    
    return model
    
    





# @ignore_warnings(category=ConvergenceWarning)
def k_fold_cross_validation(train_dir, algorithm, scaling_method,
                               val_dir = None, k = 5):
    '''Performs five fold cross validation of using the <algorithm>, 
    <scaling method>, and manually-scored data in <train_dir> and, optionally,
    <val_dir>. This is a streamlined version of earlier functions that tested
    several models and scaling methods at once, and evaluated nictation
    metrics as well as accuracy. This version only calculates the accuracy of
    the raw classifier output. Censored frames and flagged centerlines are not
    considered.'''
    
    fps = 5
    
    def prep_scores(data_dir):
        
        # load training / testing manual scores and features
        feature_file = data_dir + '//nictation_features.csv'
        df = pd.read_csv(feature_file)
        
        
        # load manual scores    
        manual_score_file =  data_dir + '//manual_nictation_scores.csv'
        man_scores_lst = load_manual_scores_csv(manual_score_file,
                                                simplify = False)
        man_scores = []
        for scr_w in man_scores_lst:
            man_scores += list(scr_w)
        df.insert(5,'manual_behavior_label',man_scores)
        
        
        # remove censored, nan/inf-containing, and flagged centerline worm-frames
        df_masked = clean_dataset(df)
        

        return df_masked
    
    df_train_test = prep_scores(train_dir)
    if val_dir != '':
        df_val = prep_scores(val_dir)

    
    # determine the group indices for x-fold cross validation
    vid_names_fold = []
    worms_fold = []
    worm_strings = []
    for i in range(len(df_train_test)):
        worm_string = df_train_test.iloc[i]['vid_name']+\
            str(df_train_test.iloc[i]['worm'])
        if worm_string not in worm_strings:
            worm_strings.append(worm_string)
            vid_names_fold.append(df_train_test.iloc[i]['vid_name'])
            worms_fold.append(df_train_test.iloc[i]['worm'])
    del worm_strings

    fold_worms = [list(np.arange(round(i*(len(worms_fold)-1)/k),
              round((i+1)*(len(worms_fold)-1)/k))) for i in range(k)]
    
    
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    
    
    fold_inds = []
    for f in range(len(fold_worms)):
        fi = []
        for w in fold_worms[f]:
            i_vid = np.where(df_train_test["vid_name"]==vid_names_fold[w])[0] 
            i_wor =  np.where(df_train_test["worm"]==worms_fold[w])[0]
            fi = fi + intersection(i_vid,i_wor)
        fold_inds.append(fi)
    
   
    # k fold cross validation
    accs_train = []; accs_test = []; accs_val = []
    for fold in range(len(fold_inds)):
        print(f"{k} fold cross validation, fold {fold+1}")
        # separate training and test sets
        df_train = df_train_test.drop(df_train_test.iloc[fold_inds[fold]].index)
        df_test = df_train_test.iloc[fold_inds[fold]]                    
        

        # scale data
        df_train_scaled, scaler = scale_training_features(df_train, 
                                    scaling_method, df_train.columns[6:])
        df_test_scaled = copy.copy(df_test)
        df_test_scaled[df_test_scaled.columns[6:]] = scaler.transform(
            df_test_scaled[df_test_scaled.columns[6:]])
        
        
        # train classifier
        model = fit_model(df_train_scaled[df_train_scaled.columns[6:]], \
            df_train_scaled['manual_behavior_label'],algorithm)
        
            
        # test classfier on the training and testing data
        probs_train = model.predict_proba(df_train_scaled[df_train_scaled.columns[6:]])
        probs_train_smooth = smooth_probabilities(probs_train,0,fps)
        preds_train = probabilities_to_predictions(probs_train,[0,1])
        accs_train.append(compare_scores(
                        df_train['manual_behavior_label'],preds_train))
        
        
        probs_test = model.predict_proba(df_test_scaled[df_test_scaled.columns[6:]])
        probs_test_smooth = smooth_probabilities(probs_test,0,fps)
        preds_test = probabilities_to_predictions(probs_test,[0,1])
        accs_test.append(compare_scores(
                        df_test['manual_behavior_label'],preds_test))
        
        # if provided, test classifier on validation data
        if val_dir != '':
            df_val_scaled = copy.copy(df_val)
            df_val_scaled[df_val_scaled.columns[6:]] = scaler.transform(
            df_val_scaled[df_val_scaled.columns[6:]])
            probs_val = model.predict_proba(df_val_scaled[df_val_scaled.columns[6:]])
            probs_val_smooth = smooth_probabilities(probs_val,0,fps)
            preds_val = probabilities_to_predictions(probs_val,[0,1])
            accs_val.append(compare_scores(
                            df_val['manual_behavior_label'],preds_val))
        else:
           accs_val.append(np.nan)
        
    print(f"{k} fold cross validation finished, saving results")
    # save a summary of the results
    d = {'fold': np.arange(0,k), 'train acc': accs_train,
         'test acc': accs_test, 'val acc' : accs_val}
    df_accs = pd.DataFrame(data=d)
    
    return df_accs



# deprecated
# @ignore_warnings(category=ConvergenceWarning)
def evaluate_models_x_fold_cross_val(vid_file_train, vid_file_test, 
                                     **kwargs):
    '''Performs x-fold cross validation of the specified models with the
    specified smoothing sigmas usin <vid_file_train> and also evaluates
    model performance on <vid_file_test>'''

    x = kwargs.get('x',5)

    model_types = kwargs.get('model_types',['logistic regression',
        'decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes',
        'support vector machine', 'random forest', 'neural network'])
    
    scaling_methods = kwargs.get('scaling_methods',['none','min max',
                        'variance','Gaussian','whiten'])
    
    exclude_censored = kwargs.get('exclude_censored',False)
    
    exclude_unfixed_centerlines = kwargs.get('exclude_unfixed_cl',False)
    
    sigmas = kwargs.get('sigmas', np.arange(0,1.5,0.1))
    
    fps = kwargs.get('fps', 5)
        
    # if a save file is provided, information from that file is loaded and
    # progress
    save_file = kwargs.get('save_file', None)
    
    # load features and manual scores for the training and test videos
    df_train, i_naninf_train = combine_and_prepare_man_scores_and_features(vid_file_train)
    df_test, i_naninf_test = combine_and_prepare_man_scores_and_features(vid_file_test)
    
    
    # # plot the abundance of worm-frames with each behavior label
    # nict_plot.bar_count(df_train)
    
    accs = np.empty((len(scaling_methods),x,len(model_types),len(sigmas)
                    ,3))
    accs[:] = np.nan
    
    times = np.empty((len(scaling_methods),x,len(model_types),2))
    times[:] = np.nan
    
    NRs = copy.copy(accs)
    IRs = copy.copy(accs)
    SRs = copy.copy(accs)
    
    
    # list of frame indices to remove
    
    
    # add censored frames and frames from the same worm in close proximity to
    # the list
    

    # remove flagged centerline frames and frames in close proximity
    # 0: never flagged, 1: flagged and not fixed, 2: flagged and fixed, 
    # 3: flagged and fixed but still meets flagging criteria. remove 1 and 3
    

    
    # removal of worm frames with or nearby censored frames and flagged
    # centerlines
    ix_cen_train = find_censored_inds(df_train)
    ix_cen_test = find_censored_inds(df_test)
    
    cln_flags_train = prepare_centerline_flags(vid_file_train, i_naninf_train)
    ix_flg_train = find_unfixed_centerline_inds(df_train, cln_flags_train)    
    cln_flags_test = prepare_centerline_flags(vid_file_test, i_naninf_test)
    ix_flg_test = find_unfixed_centerline_inds(df_test, cln_flags_test)
    
    ix_train = np.sort(np.unique(np.array(
        list(ix_cen_train)+list(ix_flg_train))))
    df_train = df_train.drop(index = ix_train)
    
    ix_test = np.sort(np.unique(np.array(
        list(ix_cen_test)+list(ix_flg_test))))
    df_test = df_test.drop(index = ix_test)
    
    df_train = df_train.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)
    
    man_scores_test_vid = separate_list_by_worm(
        df_test['manual_behavior_label'], df_test)
    
    all_worms = np.sort(np.unique(df_train['worm']))
    
    # determine the group indices for x-fold cross validation
    fold_inds = [list(np.arange(round(i*(len(all_worms)-1)/x),
              round((i+1)*(len(all_worms)-1)/x))) for i in range(x)]
    
    # actual worm numbers in case worms were removed entirely above
    fold_inds2 = copy.copy(fold_inds)
    for f in range(len(fold_inds)):
        fold_inds2[f] = all_worms[fold_inds[f]]
        
    # calculate some manual metric values
    man_scores_train_vid = separate_list_by_worm(
        df_train['manual_behavior_label'], df_train)
    man_metrics = {
        "NR_test": nict_met.nictation_ratio(man_scores_test_vid, False),
        "IR_test": nict_met.initiation_rate(man_scores_test_vid, False),
        "TR_test": nict_met.stopping_rate(man_scores_test_vid, False),
        "NR_train": nict_met.nictation_ratio(man_scores_train_vid , False),
        "IR_train": nict_met.initiation_rate(man_scores_train_vid , False),
        "TR_train": nict_met.stopping_rate(man_scores_train_vid , False)
        }

    # before starting, try loading and see if any progress as been made
    if save_file is not None and os.path.isfile(save_file):
        with open(save_file, 'rb') as f:     
            model_types, scaling_methods, sigmas, accs, times, NRs, IRs,\
                SRs, man_metrics, scaling_progress = pickle.load(f)
    else:
        scaling_progress = 0

    # begin x-fold cross validation loop
    for sm in range(scaling_progress,len(scaling_methods)):
        
        print('On scaling method '+str(sm+1)+' of '+str(len(scaling_methods)))
        if scaling_methods[sm] != 'none':
            df_train_scaled, scaler = scale_training_features(df_train, 
                                    scaling_methods[sm], df_train.columns[3:])
            
            cols = df_test.columns[3:]
            df_test_scaled = copy.deepcopy(df_test)
            df_test_scaled[cols] = scaler.transform(df_test[cols])
            
        else:
            df_train_scaled = copy.deepcopy(df_train)
            df_test_scaled = copy.deepcopy(df_test)
        
        for t in range(x):
            print('On fold ' + str(t+1) + ' of ' + str(x))

            ####
            
            w_val = list(fold_inds[t])
            w_train = []
            for wv in range(x):
                if wv != t:
                    w_train = w_train + list(fold_inds2[wv])
                    
            
            i_train = []
            for w in w_train:
                i_train = i_train + df_train_scaled.index[
                    df_train_scaled['worm'] == w].tolist()
            
            i_val = []
            for w in w_val:
                i_val = i_val + df_train_scaled.index[
                    df_train_scaled['worm'] == w].tolist()
            
            x_train, x_val, y_train, y_val, wi_train, wi_val = split_by_w_ind(
                                              df_train_scaled, i_train, i_val)
            del wv, w
            
            
            
            # Also separate the manual scores into training and validation parts
            ms_train = df_train_scaled['manual_behavior_label'].iloc[i_train]
            ms_val = df_train_scaled['manual_behavior_label'].iloc[i_val]
            ms_test = df_test_scaled['manual_behavior_label']

            
            
            # [np.array(ms_train_all[x]) for x in w_val]
            # ms_train = [ms_train_all[x] for x in w_train]
            
            ####
            # x_train, x_test, y_train, y_test, wi_train, wi_test = split(
            #     df_train_scaled, 0.75, rand_split)
            # man_scores_train_train = separate_list_by_worm(y_train, wi_train)
            # man_scores_train_test = separate_list_by_worm(y_test, wi_test)
            
            for mt in range(len(model_types)):
                print(model_types[mt])
                
                # train on training video and record training and test 
                # accuracy on this video and elapsed time
                t0 = time.time()
                categories = np.unique(y_train) # sometimes a label is not
                # represented in the training set
                mod, train_acc, train_test_acc, probs, preds = \
                    learn_and_predict(x_train, x_val, y_train, y_val,
                                      model_types[mt])
                train_time = time.time()-t0
                
                
                # make probability infrerence on the training and test 
                # portions of the training video to be used below
                probs_train_train = mod.predict_proba(x_train)
                probs_train_val = mod.predict_proba(x_val)
                
                # use the same model to make inferences on another video and
                # record the elapsed time and accuracy
                t0 = time.time()
                probs_test_vid = mod.predict_proba(
                    df_test_scaled[df_test_scaled.columns[4:]])
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
                
                    ms_train_list = separate_list_by_worm(
                            ms_train, wi_train)
                
                    NRs[sm,t,mt,sg,0] = nict_met.nictation_ratio(
                        preds_smooth_list, False)
                    IRs[sm,t,mt,sg,0] = nict_met.initiation_rate(
                        preds_smooth_list, False)
                    SRs[sm,t,mt,sg,0] = nict_met.stopping_rate(
                        preds_smooth_list, False)
                    accs[sm,t,mt,sg,0] = compare_scores_list(
                        ms_train_list,preds_smooth_list)
                    
                    
                    # smooth and inferences and calculate nictation metrics on
                    # the test set of the training video
                    probs_smooth = smooth_probabilities(
                        probs_train_val, sigmas[sg], fps)
                    preds_smooth = probabilities_to_predictions(probs_smooth,
                                                                categories)
                    preds_smooth_list = separate_list_by_worm(
                        preds_smooth, wi_val)
                    
                    ms_val_list = separate_list_by_worm(
                        ms_val, wi_val)
                
                    NRs[sm,t,mt,sg,1] = nict_met.nictation_ratio(
                        preds_smooth_list, False)
                    IRs[sm,t,mt,sg,1] = nict_met.initiation_rate(
                        preds_smooth_list, False)
                    SRs[sm,t,mt,sg,1] = nict_met.stopping_rate(
                        preds_smooth_list, False)
                    accs[sm,t,mt,sg,1] = compare_scores_list(
                        ms_val_list, preds_smooth_list)
                    
                    
                    # smooth and inferences and calculate nictation metrics on
                    # the separate test video
                    
                    probs_smooth = smooth_probabilities(
                        probs_test_vid, sigmas[sg], fps)
                    preds_smooth = probabilities_to_predictions(probs_smooth,
                                                                categories)
                    preds_smooth_list = separate_list_by_worm(
                        preds_smooth, df_test_scaled)
                
                    ms_test_list = separate_list_by_worm(
                        ms_test, df_test_scaled)
                
                    NRs[sm,t,mt,sg,2] = nict_met.nictation_ratio(
                        preds_smooth_list, False)
                    IRs[sm,t,mt,sg,2] = nict_met.initiation_rate(
                        preds_smooth_list, False)
                    SRs[sm,t,mt,sg,2] = nict_met.stopping_rate(
                        preds_smooth_list, False)
                    accs[sm,t,mt,sg,2] = compare_scores_list(
                        man_scores_test_vid, preds_smooth_list)
               
                
                # training and inference times do not include the smoothing
                # time, which is basically negligable
                times[sm,t,mt,0] = train_time
                times[sm,t,mt,1] = infr_time

        scaling_progress = sm+1
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump([model_types, scaling_methods, sigmas, accs, 
                             times, NRs, IRs, SRs, man_metrics, 
                             scaling_progress], f)   
          
    
    return accs, times, NRs, IRs, SRs, man_metrics
    
    

def find_censored_inds(df):
        '''finds indices of censored and nearby frames close enough to affect
        feature values'''
        df = df.reset_index(drop = True)
        ix_cen_prime = df.index[
            df['manual_behavior_label'] == -1].tolist()
        ix_cen_adj = []
        for i in ix_cen_prime:
            for offset in [-2,-1,1,2]:
                if i+offset > 0 and i+offset < len(df) and \
                    df.iloc[i]['worm'] == df.iloc[i+offset]['worm']:
                    ix_cen_adj.append(i+offset)
        return np.sort(np.unique(np.array(ix_cen_prime+ix_cen_adj)))
    

def find_unfixed_centerline_inds(df,flgs):
    '''finds indices of centerlines still flagged after fixing and nearby
    frames close enough to affect feature values'''
    df = df.reset_index(drop = True)
    ix_prime = list(np.where(flgs==3)[0])
    ix_adj = []
    for i in ix_prime:
        for offset in [-2,-1,1,2]:
            if i+offset > 0 and i+offset < len(df) and \
                df.iloc[i]['worm'] == df.iloc[i+offset]['worm']:
                ix_adj.append(i+offset)
    return np.sort(np.unique(np.array(ix_prime+ix_adj)))

def prepare_centerline_flags(vid_file, ix):
        '''Loads centerline flags, linearizes them, and removes flags with the
        indices ix'''
        clns, cln_flags = dmm.load_centerlines_csv(
            os.path.splitext(vid_file)[0] + \
                r'_tracking\centerlines')
        cln_flags_linear = []
        for i in range(len(cln_flags)):
            cln_flags_linear = cln_flags_linear + cln_flags[i] 
        cln_flags_linear = np.array(cln_flags_linear)
        return np.delete(cln_flags_linear,ix,0)



def separate_list_by_worm(lst, df):
    '''Takes a continuous list <lst> and splits it into a list of lists 
    based on the worm numbers in <df>'''
    
    lst = np.array(lst)
    lst_by_worm = []
    df_zeroi = df.reset_index(drop = True)
    
    #num_w =  int(df.loc[df['worm'].idxmax()][0]) + 1
    for w in np.unique(df['worm']):
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
        comp_scores[wt] = np.array(comp_scores[wt]).astype(np.int16)
        man_scores[wt] = np.array(man_scores[wt]).astype(np.int16)
        same += np.sum(comp_scores[wt]==man_scores[wt])
    return same / total
    


def compare_scores(man_scores, comp_scores):
    '''Takes two lists of worm track scores, compares them, and
    returns the overall accuracy.'''
    
    total = len(comp_scores)
    comp_scores = np.array(comp_scores).astype(np.int16)
    man_scores = np.array(man_scores).astype(np.int16)
    same = np.sum(comp_scores==man_scores)
    
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


def split_by_w_ind(df, ind_train, ind_val):
    '''Splits the data in df into training and validation X (features), y 
    (manual scores), and wi (worm number and frame) based on the indices in 
    <ind_train> and <ind_val>'''

    
    X_train = df[df.columns[4:]].iloc[ind_train]
    y_train = df['manual_behavior_label'].iloc[ind_train]
    wi_train = df[df.columns[0:3]].iloc[ind_train]
    
    X_val = df[df.columns[4:]].iloc[ind_val]
    y_val = df['manual_behavior_label'].iloc[ind_val]
    wi_val = df[df.columns[0:3]].iloc[ind_val]
    
    return X_train, X_val, y_train, y_val, wi_train, wi_val


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






# deprecated
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
    df_masked, ix = nan_inf_mask_dataframe(df)
    
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

def probabilities_to_predictions_w(probs, categories):
    '''This version will replace the version above, it assumes that the 
    probabilities are a list of lists or arrays with each sub list 
    corresponding to a worm track'''
    
    preds_all = []
    
    for w in range(len(probs)):
        pred_nums = list()
        
        for f in probs[w]: pred_nums.append(np.argmax(f,axis = -1))
        
        preds = list()
        
        for i in pred_nums: preds.append(categories[i])

        preds_all.append(preds)
    
    return preds_all#np.int8(preds)
    

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
    feats = copy.copy(df.columns[4:])

    
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
    
    df = copy.deepcopy(dataframe).reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # also return indices of removed rows
    inds = list(df[df.isna().any(axis=1)].index)
    
    df = df.dropna().reset_index(drop=True)
    
    return df, inds
    


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
def learn_and_predict(X_train, X_val, y_train, y_val,
                      model_type = 'random forest', print_acc = False):
    
    if model_type == 'logistic regression':
        model = LogisticRegression(max_iter = 1000, random_state = 0)
    elif model_type == 'decision tree':
        model = DecisionTreeClassifier(random_state = 0)
    elif model_type == 'k nearest neighbors':
        model = KNeighborsClassifier()
    elif model_type == 'linear discriminant analysis':
        model = LinearDiscriminantAnalysis()
    elif model_type == 'Gaussian naive Bayes':
        try:
            model = GaussianNB(random_state = 0)
        except:
            model = GaussianNB()
            print('GNB has no random_state')
    elif model_type == 'support vector machine':
        try:
            model = SVC(probability = True,random_state = 0)
        except:
            model = SVC(probability = True)
            print('SVC has no random_state')
        #print('WARNING: SVM probabilities may not correspond to scores.')
    elif model_type == 'random forest':
        try:
            model = RandomForestClassifier(max_features='sqrt', random_state = 0)
        except:
            model = RandomForestClassifier(max_features='sqrt')
            print('Random Forest has no random_state')

    elif model_type == 'neural network':
        model = MLPClassifier(random_state = 0)
    else:
        print('WARNING: model type "'+model_type+'" not recognized!')
    
    model.fit(X_train, y_train)
    
    if print_acc:
        print('Accuracy of ',model_type,' classifier on training set: {:.2f}'
             .format(model.score(X_train, y_train)))
        
        if len(X_val) > 0:
            print('Accuracy of ',model_type,' classifier on validation set: {:.2f}'
                 .format(model.score(X_val, y_val)))
    
    train_acc = model.score(X_train, y_train)
    if len(X_val) > 0:
        val_acc = model.score(X_val, y_val)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)
    
    if len(X_val) > 0:
        return model, train_acc, val_acc, probabilities, predictions
    else:
        return model, train_acc





def calculate_features(vid_file):
    
        
    gap = 1
    halfwidth = 88
    path_f = 3
    
    
    # load centroids, first frames, centerlines, and centerline flags
    cents, ffs  = dmm.load_centroids_csv(
        os.path.splitext(vid_file)[0] + \
            r'_tracking\centroids.csv')
    
    clns, cln_flags = dmm.load_centerlines_csv(
        os.path.splitext(vid_file)[0] + \
            r'_tracking\centerlines')

    
    # load tracking parameters
    params = dmm.load_parameter_csv(
        os.path.splitext(vid_file)[0] +\
            r'_tracking\tracking_parameters.csv')
    
    
    vid = cv2.VideoCapture(vid_file)
    bkgnd = trkr.Tracker.get_background(vid,10) # needed?
    

    
    # names of features
    cols = ['worm', 'frame', 'video_frame','centerline_flag', 'blur',
            'bkgnd_sub_blur', 'bkgnd_sub_blur_ends','ends_mov_bias', 
            'body_length', 'total_curvature','lateral_movement',
            'longitudinal_movement', 'out_of_track_centerline_mov',
            'angular_sweep', 'cent_path_past', 'cent_path_fut', 
            'centroid_progress', 'head_tail_path_bias',
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
                        'video_frame' : int(f + ffs[w]),
                        'centerline_flag' : int(cln_flags[w][f])
                        }
            
            
            df = pd.concat([df, pd.DataFrame(new_row, index = [0])], axis = 0,
                           ignore_index=True)

            cl0 = copy.copy(cl)
    
    
    # tack on activity
    df.insert(4,'activity',activity)
    
    
    # calculate first derivatives
    fps = vid.get(cv2.CAP_PROP_FPS)
    df = calculate_metafeatures(df,fps)
    
    
    # save feature values
    df.to_csv(os.path.splitext(vid_file)[0] + \
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
    df_masked = nan_inf_mask_dataframe(df)[0]
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
        
        train_dir = r'E:\behavior_training_Celegans_vid_cropped_scaled'
        algorithm = 'random forest'
        scaling_method = 'whiten'
        val_dir = ''
 
        
        results =  k_fold_cross_validation(train_dir, algorithm, 
                                           scaling_method, val_dir, 5)
        # vid_dir = r"D:\Data_flp_7_updated"
        # file_list = sorted(os.listdir(vid_dir))
        # for f in file_list[:]:
        #     if f[-4:] == '.avi' and f[:-4]+'_tracking' in file_list:
        #         calculate_features(vid_dir + '\\' + f)
        

        
    except:
        
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


