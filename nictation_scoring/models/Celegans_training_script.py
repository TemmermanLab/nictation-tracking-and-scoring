# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:41:28 2022

This script trains a model for scoring C. elegans nictation based on both
available manually-scored videos

@author: PDMcClanahan
"""
import numpy as np
import pandas as pd
import pickle
import csv
import copy
import os
import sys
import matplotlib.pyplot as plt

from pathlib import Path
home = str(Path.home())

sys.path.append(home + '//Dropbox//Temmerman_Lab//code//tracking-and-' + \
                'scoring-nictation//nictation_scoring')

import nictation_module as nm
import nictation_plotting as nict_plt
import nictation_metrics as nict_met

fps = 5
vid_file_train = "D://Celegans_nictation_dataset//Ce_R3_d06.avi"
vid_file_test = "D://Celegans_nictation_dataset//Ce_R2_d21.avi"


model_type = 'random forest'
scaling_method = 'whiten'
prop_for_training = 1.0
rand_split = False
only_active = False
fps = 5
binary = True

model_file = 'C:\\Users\\PDMcClanahan\\Dropbox\\Temmerman_Lab\\code\\' + \
    'tracking-and-scoring-nictation\\nictation_scoring\\models\\' + \
     'Ce_random_forest_and_whiten_scaler_trained_on_both_videos.pkl'



if not os.path.exists(model_file):
    print('training and saving model')
    # load, mask, scale, and split training data
    df_train = nm.combine_and_prepare_man_scores_and_features(vid_file_train)
    df_test = nm.combine_and_prepare_man_scores_and_features(vid_file_test)
    df_combined = pd.concat([df_train, df_test], axis=0)
    df_scaled, scaler = nm.scale_training_features(df_combined,
                                      scaling_method, df_combined.columns[4:])
    x_train, x_test, y_train, y_test, wi_train, worminf_test = nm.split(
                    df_scaled, prop_for_training, rand_split)
    
    # train a model
    mod, train_acc = nm.learn_and_predict(
                        x_train, x_test, y_train, y_test, model_type)
    
    with open(model_file, 'wb') as f:
        pickle.dump([mod, scaler], f)
    
else:
    print('model file already exists, loading model')
    with open(model_file, 'rb') as f: 
        mod, scaler = pickle.load(f)








