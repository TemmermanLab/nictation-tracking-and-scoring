# -*- coding: utf-8 -*-
"""
Created on Thur Feb 09 19:51:05 2023

This script uses 100% of the training and test datasets to train a neural
network which is then used to score the remainder of the C. elegans timecourse
dataset. It uses min max scaling.

@author: PDMcClanahan
"""

import numpy as np
import pandas as pd
import pickle
import copy
import os
import sys

from sklearn.neural_network import MLPClassifier

from pathlib import Path
home = str(Path.home())

sys.path.append(home + '//Dropbox//Temmerman_Lab//code//tracking-and-' + \
                'scoring-nictation//nictation_scoring')

import nictation_module as nm
import data_management_module as dmm


vid_file_train = "E://Celegans_nictation_dataset//Ce_R3_d06.avi"
vid_file_test = "E://Celegans_nictation_dataset//Ce_R2_d21.avi"
# model_type = 'neural network'
scaling_method = 'min max' # or 'none'


model_file = 'C:\\Users\\PDMcClanahan\\Dropbox\\Temmerman_Lab\\code\\' + \
    'nictation-scoring-paper-analysis\\Celegans_timecourse\\' + \
    'neural_network_and_min_max_train_and_test.pkl'
    
    
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
            

# remove flagged centerline frames and frames in close proximity
# 0: never flagged, 1: flagged and not fixed, 2: flagged and fixed, 
# 3: flagged and fixed but still meets flagging criteria. remove 1 and 3
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
    
    

if not os.path.exists(model_file):
    
    df_train, i_naninf_train = nm.combine_and_prepare_man_scores_and_features(
        vid_file_train,None,True,'nictation_features.csv')
    df_test, i_naninf_test = nm.combine_and_prepare_man_scores_and_features(
        vid_file_test,None,True,'nictation_features.csv')
    
    # check that the number of nan/inf worm frames is similar
    
    # continue
    ix_cen_train = find_censored_inds(df_train)
    ix_cen_test = find_censored_inds(df_test)
    
    cln_flags_train = prepare_centerline_flags(vid_file_train, i_naninf_train)
    ix_flg_train = find_unfixed_centerline_inds(df_train, cln_flags_train)    
    cln_flags_test = prepare_centerline_flags(vid_file_test, i_naninf_test)
    ix_flg_test = find_unfixed_centerline_inds(df_test, cln_flags_test)
    
    # drop the censored and unfixed centerlines
    ix_train = np.sort(np.unique(np.array(
        list(ix_cen_train)+list(ix_flg_train))))
    df_train = df_train.drop(index = ix_train)
    
    ix_test = np.sort(np.unique(np.array(
        list(ix_cen_test)+list(ix_flg_test))))
    df_test = df_test.drop(index = ix_test)
    
    df_train = df_train.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)
    
    df = pd.concat([df_train, df_test], axis=0)
    
    # scale the features and train a model using the fixed centerlines
    if scaling_method != 'none':
        df_scaled, scaler = nm.scale_training_features(df,scaling_method,
                                                       df.columns[4:])         
        cols = df.columns[4:]
    else:
        df_scaled = copy.deepcopy(df)                            
                            
    X = df_scaled[df_scaled.columns[4:]]
    ms = df_scaled['manual_behavior_label']
    
    model = MLPClassifier(random_state = 0)
    print('training model, can take awhile...')
    model.fit(X,ms)

    with open(model_file, 'wb') as f:
        pickle.dump([model, scaler], f)
        
        
else:
    print('model file already exists, loading model')
    with open(model_file, 'rb') as f: 
        mod, scale = pickle.load(f)











