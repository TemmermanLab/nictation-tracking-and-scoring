# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:50:59 2022


This script uses the original video and tracking output (centroids, 
centerlines, and scale) to calculate features for detecting nictation and 
saves those features for later use.


Inputs:
    1. original video file (tracking output located relative to this)

    
Outputs:
    1. nictation feature values saved as a .csv


Dependencies:
    1. nictation_features.py (in same directory)
    2. tracker_classes.py
    

This script is based on 1_calculate_indicators.py, which itself was base on
the older CNI_main.py


Issues and improvements:
    -the halfwidth should be standardized based on the worm type and scale
    -blur might benefit from background subtraction to minimize the effect of
     the pillars
    -it might be nicer if calculating the nictation features was one call that
     would take all the necessary inputs and return the features and their
     names; then changing or adding metrics would be only a matter of editing
     the nictation_features module
    -there are hard-coded parameters in calculating diff img activity
    -features that require loading a video frame (blur and diff img activity)
     could be combined to minimize the number of times video frames are read
    

@author: PDMcClanahan
"""

import pickle
import numpy as np
import sys
import cv2
import os
import pandas as pd
import copy

sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictat'+\
    r'ion\nictation_scoring_training\training')

try:
    sys.path.append(os.path.split(__file__)[0])
except:
    pass


sys.path.append(
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation_20220523')
sys.path.append(
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation_20220523\nictation_scoring_training\training')
import nictation_features as nf
import nict_scoring_functions as nsf
import tracker_classes as trkr

def calculate_features(vid_file):
    
    
    gap = 1
    halfwidth = 88
    path_f = 3
    
    # load centroids, first frames, centerlines, and centerline flags
    cents, ffs  = trkr.Tracker.load_centroids_csv(
        os.path.splitext(vid_file)[0] + r'_tracking\centroids.csv')
    
    clns, cln_flags = trkr.Tracker.load_centerlines_csv(
        os.path.splitext(vid_file)[0] + r'_tracking\centerlines')

    
    # load tracking parameters
    params = trkr.Tracker.load_parameter_csv_stat(
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
        print('Calculating features for worm ' + str(w) + ' of ' + 
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
    df = nsf.first_derivative_df(df,fps)
    
    # save indicator values
    df.to_csv(os.path.splitext(vid_file)[0] + 
              r'_tracking\nictation_features.csv', index = False)
    



# testing
if __name__ == '__main__':
    try:
        
        # vf = "C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictation_dataset"+\
        #     "\\Ce_R2_d21.avi"
        
        # calculate_features(vf)
        
        vid_dir = r"C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictation_dataset"
        file_list = os.listdir(vid_dir)
        for f in file_list[56:]:
            if f[-4:] == '.avi' and f[:-4]+'_tracking' in file_list:
                calculate_features(vid_dir + '\\' + f)
        
        
        # vf = r"C:\Users\Temmerman Lab\Desktop\test_data_for_tracking\R1d4_first_four.avi"
        # calculate_features(vf)
        
        
    except:
        
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)