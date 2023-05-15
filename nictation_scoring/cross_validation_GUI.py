# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:55:45 2023

@author: PDMcClanahan
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
import copy
import pandas as pd

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

from pathlib import Path
home = str(Path.home())

sys.path.append(home + '//Dropbox//Temmerman_Lab//code//tracking-and-' + \
                'scoring-nictation//nictation_scoring')

os.chdir(home + '//Dropbox//Temmerman_Lab//code//nictation-scoring-paper-' + \
         'analysis//Celegans_timecourse')

import nictation_module as nm
import nictation_metrics as nict_met


run_cross_val = False
var_file = 'C_elegans_full_cross_val_var_smoothing_and_metric_accuracy.pkl'


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
    
    exclude_censored = kwargs.get('exclude_censored',True)
    
    exclude_unfixed_centerlines = kwargs.get('exclude_unfixed_cl',True)
    
    sigmas = kwargs.get('sigmas', 0)
    
    fps = kwargs.get('fps', 5)
    
    import pdb; pdb.set_trace()
    # if a save file is provided, information from that file is loaded and
    # progress
    save_file = kwargs.get('save_file', None)
    
    # load features and manual scores for the training and test videos
    df_train, i_naninf_train = combine_and_prepare_man_scores_and_features(
        vid_file_train)
    df_test, i_naninf_test = combine_and_prepare_man_scores_and_features(
        vid_file_test)
    
    
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


def write_summary_csv():
    dir_train = train_entry.get()
    dir_val = val_entry.get()
    algorithm = algo_menu.get()
    
    evaluate_models_x_fold_cross_val(dir_train, dir_val, \
                                     model_types = algorithm)
    
def cross_validation_GUI():
    
    train_dir = []
    test_dir = []
    algorithm = []
    
    
    xval_GUI = tk.Tk()
    
    train_label = Label(xval_GUI,text="Training dataset:")
    train_label.grid(row=0, column=0, columnspan = 1,sticky="w")
    train_entry = Entry(xval_GUI)
    train_entry.grid(row=0, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    val_label = Label(xval_GUI,text="Validation dataset:")
    val_label.grid(row=1, column=0, columnspan = 1,sticky="w")
    val_entry = Entry(xval_GUI,text="")
    val_entry.grid(row=1, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    
    scale_meth_label = Label(xval_GUI,text="Scaling method:")
    scale_meth_label.grid(row=2, column=0, columnspan = 1,sticky="w")
    scaling_methods = ['none','min max','variance','Gaussian','whiten']
    scale_meth_strv = StringVar()
    scale_meth_strv.set( "none" )
    scale_meth_drop = OptionMenu(xval_GUI , scale_meth_strv , *scaling_methods) \
    .grid(row = 2,column = 1,columnspan=3,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
    alg_label = Label(xval_GUI,text="Algorithm:")
    alg_label.grid(row=3, column=0, columnspan = 1,sticky="w")
    algorithms =['logistic regression','decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes',
        'support vector machine', 'random forest', 'neural network']
    alg_strv = StringVar()
    alg_strv.set( "random forest" )
    algorithm_drop = OptionMenu(xval_GUI , alg_strv , *algorithms) \
    .grid(row = 3,column = 1,columnspan=3,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
        
    def select_training_set_button():
        root = tk.Tk()
        train_dir = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing features and manual scores for training... \
                ...")
        root.destroy()
        train_entry.delete(0, tk.END)
        train_entry.insert(0, train_dir)
        
    
    def select_validation_set_button():
        root = tk.Tk()
        val_set_loc = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing features and manual scores for validation... \
                ...")
        root.destroy()
        val_entry.delete(0, tk.END)
        val_entry.insert(0, text = val_set_loc)
        
    
    def run_button():
        train_dir = train_entry.get()
        val_dir = train_entry.get()
        scale_meth = scale_meth_strv.get()
        alg = alg_strv.get()
        accs, times, NRs, IRs, SRs, man_metrics = \
            evaluate_models_x_fold_cross_val(train_dir, val_dir,
                                model_types = alg, scale_methods = scale_meth)
                                    
        
        
    
         
    def exit_button():
        xval_GUI.destroy()
        xval_GUI.quit()
    
    
    
    tk.Button(xval_GUI,
              text = "SELECT TRAINING SET",
              command = select_training_set_button) \
              .grid(row = 4,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(xval_GUI,
              text = "SELECT VALIDATION SET",
              command = select_validation_set_button) \
              .grid(row = 4,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    tk.Button(xval_GUI,
              text = "RUN",
              command = run_button) \
              .grid(row = 4,
                    column = 2,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(xval_GUI,
              text = "EXIT",
              command = exit_button) \
              .grid(row = 4,
                    column = 3,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    xval_GUI.title('Cross Validation GUI')
    
    curr_row = 0
    
    mainloop()


if __name__ == '__main__':
    
    try:
        cross_validation_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)