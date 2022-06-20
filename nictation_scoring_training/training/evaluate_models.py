# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:51:34 2022

This script loads features and manual scores from a training / test set and
evaluates the performance of several machine learning models.  It repeats this
evaluation with different forms of data normalization and with shuffled or 
split data.

Issues and improvements:
    -Manual nictation scores here and in the nictation scoring GUI are saved
     and re-loaded from a pickle file
    -Calculate and use first derivatives as well

@author: PDMcClanahan
"""

import pandas as pd
import os
import pickle
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    sys.path.append(os.path.split(__file__)[0])
except:
    pass
import nict_scoring_functions as nsf
import nictation_metrics as nm
import plotting as pltg



def evaluate_models_accuracy(vid_file):
    '''Trains and tests several types of machine learning algorithms based on
    the features calculated from and manual scores provided for <vid_file>
    split 75 / 25 training / testing.  The resulting accuracies of the 
    different models with different types of feature normalization are 
    displayed and saved in a heat map.'''   
    
    # load manual scores
    # man_scores_sep = pickle.load( open(os.path.splitext(vid_file)[0] + \
    #                            r'_tracking/manual_nictation_scores.p','rb'))
    man_scores_lst = nsf.load_manual_scores_csv(
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
    df_masked = nsf.nan_inf_mask_dataframe(df)
    
    # split into training and test groups
    scaling_methods = ['none','min max','variance','Gaussian','whiten']
    
    scaling_meth_for_smoothing = 'Gaussian'
    
    model_types = ['logistic regression','decision tree',
                   'k nearest neighbors', 'linear discriminant analysis',
                   'Gaussian naive Bayes', 'support vector machine', 
                   'random forest', 'neural network']
    
    # initialize heatmap for accuracy
    heatmap_acc = np.empty((len(model_types),10))
    
    
    # plot the abundance of worm-frames with each behavior label
    pltg.bar_count(df)
    
    
    for sm in range(len(scaling_methods)):
        print(scaling_methods[sm])
        
        if scaling_methods[sm] != 'none':
            df_scaled = nsf.scale_data(
                df_masked, method = scaling_methods[sm])
        else:
            df_scaled = copy.deepcopy(df_masked)
        
        x_train, x_test, y_train, y_test, wi_train, worminf_test = nsf.split(
            df_scaled, 0.75)
        
        
        for mt in range(len(model_types)):
            print(model_types[mt])
            
            mod, train_acc, test_acc, probs, preds = nsf.learn_and_predict(
                x_train, x_test, y_train, y_test, model_types[mt])
            
            heatmap_acc[mt,2*sm:2*sm+2] = [train_acc,test_acc]
            
            
                

    

    # accuracy heat map figure
    fig, axes = plt.subplots()
    im = axes.imshow(heatmap_acc,cmap='viridis', vmin = 0.67, vmax = 1.00)
    #im = axes.imshow(heatmap_acc,cmap='viridis', vmin = 0.00, vmax = 1.00)
    plt.title('Model Performance with Scaled Features')
    axes.xaxis.set_label_position('top')
    axes.xaxis.tick_top() 
    axes.set_xticks([0,1,2,3,4,5,6,7,8,9])
    axes.set_xticklabels(['train','test','train','test','train','test',
                          'train','test','train','test'])
    axes.set_yticks(np.arange(len(model_types)))
    axes.set_yticklabels(model_types)
    axes.set_xlabel(
        '   none      min-max     variance   Gaussian   whitening ')
    plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center', 
             rotation_mode = 'anchor')
    
    for i in range(10):
        for j in range(len(model_types)):
            text = axes.text(i,j,"%0.2f" % heatmap_acc[j,i],ha='center',
                             va='center',fontweight = 'bold')
    
    plt.savefig(os.path.splitext(vid_file)[0] + \
                r'_tracking/nict_scoring_model_accuracy.png', dpi = 200)
    
    plt.show()

    

def evaluate_models_metrics(vid_file, scaling_method = 'Gaussian'):    
    '''Evualuates the effect of smoothing the probability outputs various models
    trained using the features (scaled by <scaling_method>)
    and manual scores of <vid_file> and plots the results as a heat map where
    cool colors signify too low, and warm colors signifiy too high'''
    
    # load manual scores
    man_scores_lst = nsf.load_manual_scores_csv(
        os.path.splitext(vid_file)[0] + \
        r'_tracking/manual_nictation_scores.csv')
    

    # load features
    df = pd.read_csv(os.path.splitext(vid_file)[0] + 
                      r'_tracking\nictation_features.csv')
    
    # get frame rate
    vid = cv2.VideoCapture(vid_file)
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # add manual scores to df
    man_scores = []
    for scr_w in man_scores_lst:
        man_scores += list(scr_w)
    df.insert(2,'manual_behavior_label',man_scores)

    # remove NaN values
    df_masked = nsf.nan_inf_mask_dataframe(df)
    
    # split into training and test groups 
    scaling_method = 'Gaussian'
    
    model_types = ['logistic regression','decision tree','k nearest neighbors',
              'linear discriminant analysis','Gaussian naive Bayes',
              'support vector machine','random forest', 'neural network']
    # model_types = ['k nearest neighbors','k nearest neighbors','k nearest neighbors',
    #           'k nearest neighbors','k nearest neighbors',
    #           'k nearest neighbors','k nearest neighbors', 'k nearest neighbors']
    
    
    # initialize heatmaps for smoothing
    sigmas = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0]
    heatmap_ir = np.empty((len(model_types),len(sigmas)+1))
    heatmap_sr = np.empty((len(model_types),len(sigmas)+1))
    heatmap_nr = np.empty((len(model_types),len(sigmas)+1))
    heatmap_nd = np.empty((len(model_types),len(sigmas)+1))
    
  
    

        
    if scaling_method != 'none':
        df_scaled = nsf.scale_data(
            df_masked, method = scaling_method)
    else:
        df_scaled = copy.deepcopy(df_masked)
    
    x_train, x_test, y_train, y_test, worminf_train, worminf_test = nsf.split(
        df_scaled, 0.75)
    
    
    for mt in range(len(model_types)):
        print(model_types[mt])
        
        mod, train_acc, test_acc, probs, preds = nsf.learn_and_predict(
            x_train, x_test, y_train, y_test, model_types[mt])
        
        # smoothing accuracies
        man_scores_lst_test = nsf.split_man_scores(man_scores_lst,
                                                   worminf_test)
        heatmap_ir[mt,0] = nm.initiation_rate(man_scores_lst_test,True,fps)
        heatmap_sr[mt,0] = nm.stopping_rate(man_scores_lst_test,True,fps)
        heatmap_nr[mt,0] = nm.nictation_ratio(man_scores_lst_test,True)
        heatmap_nd[mt,0],ret = nm.nictation_duration(man_scores_lst_test,
                                                     False, True, fps)
        
        for sigma in range(len(sigmas)):
            probs_smooth = nsf.smooth_probabilities(probs, sigmas[sigma], fps)
            preds = nsf.probabilities_to_predictions(probs_smooth)
            preds_wt = nsf.split_scores_by_wormtrack(preds, worminf_test)
            # generally nictation metrics need scores split up by worm track
            heatmap_ir[mt,sigma+1] = nm.initiation_rate(preds_wt,True,fps)
            heatmap_sr[mt,sigma+1] = nm.stopping_rate(preds_wt,True,fps)
            heatmap_nr[mt,sigma+1] = nm.nictation_ratio(preds_wt,True)
            heatmap_nd[mt,sigma+1],ret = nm.nictation_duration(preds_wt,
                                                            False, True, fps)
                
        el = 0
        for w in preds_wt:
            el += len(w)
        print(el)

    
    
    def norm_heatmap(heatmap):
        heatmap_norm = copy.copy(heatmap)
        for row in range(len(heatmap)):
            heatmap_norm[row] = heatmap[row] / heatmap[row][0]
        return heatmap_norm
    

    def make_heatmap(hm, ss, mts, title, save_file):
        '''Creates a heatmap figure of the values of a nictation metric 
        calculated using manual and automated scoring smoothed by different 
        amounts using <hm> for the values, smoothing sigma <ss> values for the
        x-axis (along with the values calculated from the manual scores in the
        first column), model type (mts) on the y-axis, and title <title> 
        (indicating the nictation metric shown) and saves the heatmap in 
        <save_file>'''
        
        hm_norm = norm_heatmap(hm)
        fig, axes = plt.subplots()
        im = axes.imshow(hm_norm,cmap='seismic', vmin = 0, vmax = 2.00)
        
        plt.title(title)
        axes.xaxis.set_label_position('bottom')
        axes.xaxis.tick_bottom() 
        axes.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
        axes.set_xticklabels(['manual',str(ss[0]),str(ss[1]),
                          str(ss[2]),str(ss[3]),str(ss[4]),
                          str(ss[5]),str(ss[6]),str(ss[7]),
                          str(ss[8]),str(ss[9])])
        axes.set_yticks(np.arange(len(mts)))
        axes.set_yticklabels(mts)
        axes.set_xlabel('sigma')
        plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center',
                 rotation_mode = 'anchor')
        
        for i in range(len(ss)+1):
            for j in range(len(mts)):
                text = axes.text(i,j,"%0.4f" % hm[j,i],ha='center',
                                 va='center', fontweight = 'bold', 
                                 rotation = 45, fontsize = 7)
        
        plt.savefig(save_file, dpi = 200)
        
        plt.show()
        

    make_heatmap(heatmap_nr, sigmas, model_types,
                 'Effect of Smoothing on Nictation Ratio',
                 os.path.splitext(vid_file)[0] + 
                 r'_tracking/nict_scoring_nict_ratio_smoothing.png')
    
    make_heatmap(heatmap_nd, sigmas, model_types,
                 'Effect of Smoothing on Nictation Duration',
                 os.path.splitext(vid_file)[0] + 
                 r'_tracking/nict_scoring_nict_dur_smoothing.png')
        
    make_heatmap(heatmap_sr, sigmas, model_types,
                 'Effect of Smoothing on Stopping Rate',
                 os.path.splitext(vid_file)[0] + 
                 r'_tracking/nict_scoring_stopping_rate_smoothing.png')
            
    make_heatmap(heatmap_ir, sigmas, model_types,
                 'Effect of Smoothing on Initiation Rate',
                 os.path.splitext(vid_file)[0] + 
                 r'_tracking/nict_scoring_init_rate_smoothing.png')
        
    # effect of smoothing on initiation rate
    
    # effect of smoothing on duration
    
    # effect of smoothing on stopping rate
    
    # heat map of nictation ratio with smoothing
    
    
    
    # heat map of initiation rate with smoothing
    
    
    # heat map of stopping rate with smoothing
    


# def evaluate_shuffle_vs_split():
#     pass


# testing
if __name__ == '__main__':
    try:
        
        # vf = r"C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation" +\
        #     r"\nictation_scoring_training\training\test.avi"
        vf = "C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictation_dataset"+\
            "\\Ce_R2_d21.avi"
        
        # evaluate_models_accuracy(vf)
        evaluate_models_metrics(vf)
        
    except:
        
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

