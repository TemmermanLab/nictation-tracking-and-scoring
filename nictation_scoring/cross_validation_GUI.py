# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:16:06 2023

For running five fold cross validation on a manually scored nictation dataset
specified by the user

@author: PDMcClanahan
"""
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


# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend(frameon = False,fontsize = 7)

# ax.set_xlabel('Smoothing $\sigma$ (s)',font = 'Arial',fontsize=8)
# ax.set_ylabel('Relative error',font = 'Arial',fontsize=8)

# ax.tick_params(axis="x", labelsize=7) # set_xticklabels ignores fontsize
# ax.tick_params(axis="y", labelsize=7) # set_xticklabels ignores fontsize

# fig.savefig('five_fold_val_metric_accuracy_by_smoothing_sigma_all_Ce.png',dpi = 300,
#             bbox_inches = 'tight')
# plt.show()

# print('Nictation ratio is most accurate at sigma = '+\
#       str(round(sigmas[np.where(abs(np.mean(NRs_rel_sm,0))==min(abs(np.mean(NRs_rel_sm,0))))[0][0]],3))+\
#       ' when it is '+\
#       str(round(np.mean(NRs_rel_sm,0)[np.where(abs(np.mean(NRs_rel_sm,0))==min(abs(np.mean(NRs_rel_sm,0))))[0][0]],5))+\
#       ' relative to the ground truth value.')
    
# print('Initiation rate is most accurate at sigma = '+\
#       str(round(sigmas[np.where(abs(np.mean(IRs_rel_sm,0))==min(abs(np.mean(IRs_rel_sm,0))))[0][0]],3))+\
#       ' when it is '+\
#       str(round(np.mean(IRs_rel_sm,0)[np.where(abs(np.mean(IRs_rel_sm,0))==min(abs(np.mean(IRs_rel_sm,0))))[0][0]],5))+\
#       ' relative to the ground truth value.')
    
# print('Stopping rate is most accurate at sigma = '+\
#     str(round(sigmas[np.where(abs(np.mean(SRs_rel_sm,0))==min(abs(np.mean(SRs_rel_sm,0))))[0][0]],3))+\
#     ' when it is '+\
#     str(round(np.mean(SRs_rel_sm,0)[np.where(abs(np.mean(SRs_rel_sm,0))==min(abs(np.mean(SRs_rel_sm,0))))[0][0]],5))+\
#     ' relative to the ground truth value.')

# print('Overall transition rate is most accurate at sigma = '+\
#     str(round(sigmas[np.where(abs(np.mean(TRs_rel_sm,0))==min(abs(np.mean(TRs_rel_sm,0))))[0][0]],3))+\
#     ' when it is '+\
#     str(round(np.mean(TRs_rel_sm,0)[np.where(abs(np.mean(TRs_rel_sm,0))==min(abs(np.mean(TRs_rel_sm,0))))[0][0]],5))+\
#     ' relative to the ground truth value.')


# # OVERALL ACCURACY
# lims = np.arange(0,1001,1)
# fig, ax = plt.subplots(figsize = (1.5,2))
# ax.plot(sigmas[lims],np.mean(accs_sm[:,:,0],1)[lims],'k-')

# i = np.median(np.where(np.mean(accs_sm[:,:,0],1)==max(np.mean(accs_sm[:,:,0],1))))
# print(sigmas[int(i)])
# # ax.annotate("$\sigma$ = 1.689 s\naccuracy = 0.936", xy=(1.689, 0.93562), 
# #             xytext=(1, .93), arrowprops=dict(arrowstyle="->"), 
# #             font = 'Arial', fontsize = 8)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend(frameon = False,fontsize = 7)

# ax.set_xlabel('Smoothing $\sigma$ (s)',font = 'Arial',fontsize=8)
# ax.set_ylabel('Accuracy',font = 'Arial',fontsize=8)

# # ax.set_xticks([0,.5,1])

# ax.tick_params(axis="x", labelsize=7) # set_xticklabels ignores fontsize
# ax.tick_params(axis="y", labelsize=7) # set_xticklabels ignores fontsize

# fig.savefig('five_fold_val_scoring_accuracy_by_smoothing_sigma_all_Ce.png',dpi = 300,
#             bbox_inches = 'tight')
# plt.show()

# print('Scoring accuracy is highest at sigma = '+\
#       str(round(sigmas[np.where(np.mean(accs_sm[:,:,0],1)==max(np.mean(accs_sm[:,:,0],1)))[0][0]],3))+\
#       ' when it is '+\
#       str(round(np.mean(accs_sm[:,:,0],1)[np.where(np.mean(accs_sm[:,:,0],1)==max(np.mean(accs_sm[:,:,0],1)))[0][0]],5))+\
#       '.')

    
# lims = np.arange(0,1001,1)  
# overall = (abs(np.mean(NRs_rel_sm,0)) + abs(np.mean(IRs_rel_sm,0)) + abs(np.mean(SRs_rel_sm,0)) + abs(np.mean(accs_sm[:,:,0],1)[0:1001]/1-1))
# fig, ax = plt.subplots(figsize = (1.5,2))
# ax.plot(sigmas[lims],overall[lims],'k-')


# overall[np.where(overall==min(overall))[0]]

# ax.annotate("$\sigma$ = 0.193 s\nc. r. err. = 0.095", xy=(0.193, 0.09466), 
#             xytext=(.35, .11), arrowprops=dict(arrowstyle="->"), 
#             font = 'Arial', fontsize = 8)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# ax.set_xlabel('Smoothing $\sigma$ (s)',font = 'Arial',fontsize=8)
# ax.set_ylabel('Combined relative error',font = 'Arial',fontsize=8)

# ax.tick_params(axis="x", labelsize=7) # set_xticklabels ignores fontsize
# ax.tick_params(axis="y", labelsize=7) # set_xticklabels ignores fontsize

# fig.savefig('five_fold_val_combined_overall_error_by_smoothing_sigma_all_Ce.png',dpi = 300,
#             bbox_inches = 'tight')
# plt.show()

# print('Overall error is lowest at sigma = '+\
#       str(round(sigmas[np.where(overall==min(overall))[0][0]],3))+\
#       ' when it is '+\
#       str(round(overall[np.where(overall==min(overall))[0][0]],5))+\
#       '.')




