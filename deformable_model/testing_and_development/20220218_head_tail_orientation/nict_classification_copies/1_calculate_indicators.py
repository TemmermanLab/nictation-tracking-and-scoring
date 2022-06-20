# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:43:11 2021

This script uses centroid positions, centerline positions, and the original
video of tracked dauers / IJs and uses this information to calculate
indicators of nictation

Inputs:
    1. centroids pickle file
    2. centerlines pickle file
    3. original .avi video that was tracked
    
Outputs:
    1. nictation indicator values pickle file

Dependencies:
    1. custom_indicators.py
    2. other_functions.py
    
This script is based on the older CNI_main.py

Issues and improvements:
    -

@author: PDMcClanahan
"""

# RECALCULATE CUSTOM NICTATION INDICATORS BASED ON THE CORRECTIONS MENTIONED
# IN THE 13 JULY NICTATION MTG


# modules
import pickle
import numpy as np
import sys
import cv2
import os

# custom modules 
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn')
try:
    sys.path.append(os.path.split(__file__)[0])
except:
    pass
import indicator_functions as indfuns
import other_functions as other

# load tracking data and manual scores
centroids = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\centroids_clean.p','rb'))
centerlines = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\centerlines_clean.p','rb'))
scores = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\051821_quaternary_scoring\manual_nictation_scores.p','rb'))
vid = cv2.VideoCapture(r'D:\20210212_Cu_ring_test\dauers 14-14-56.avi')
bkgnd = other.get_background(vid)

# set scale, smoothing, and other defaults
um_per_pix = 3.5
k_sz = (25,25)
k_sig = 2.0
bw_thr = 20
halfwidth = 60
#offset = 5

# names of indicators
indicators = ['ends_mov_bias','out_of_track_centerline_mov',
           'blur','total_curvature','lateral_movement','longitudinal_movement','centroid_path',
           'angular_sweep','body_length','centroid_progress',
           'centroid_path_PC_var_ratio','centroid_path_PC_var_product']


# create a list of lists of arrays to hold nictation inticator values
indicator_vals = []
for m in range(len(indicators)):
    empty_indicators = []
    for w in range(len(scores)):
        empty_scores = np.empty(len(scores[w])+1)
        empty_scores[:] = np.NaN
        empty_indicators.append(empty_scores)
    indicator_vals.append(empty_indicators)
del empty_scores,m,empty_indicators


# calculate nictation indicator values (takes awhile)
for w in range(len(centroids)):
    ff = 0 # index for saving the indicator value in indicator_vals
    
    for f in range(len(centroids[w])):
        if ~np.isnan(centroids[w,f,0]):
            print('Computing nictation indicators for worm '+str(w)+' of '+str(len(centroids))+', frame '+str(f)+'. ')
            # get centroid and centerline and and smooth centerline
            centerline1 = indfuns.smooth_centerline(np.float64(centerlines[w][ff]))
            centroid1 = centroids[w,f]
            
            # calculate indicators based on one frame
            indicator_vals[2][w][ff] = indfuns.blur(vid,w,f,centroid1,um_per_pix,bw_thr,k_sz,k_sig,bkgnd,halfwidth)
            indicator_vals[3][w][ff] = indfuns.total_curvature(centerline1)
            indicator_vals[6][w][ff] = indfuns.centroid_path(w,f,centroids,um_per_pix)
            indicator_vals[8][w][ff] = indfuns.body_length(centerline1,um_per_pix)
            indicator_vals[9][w][ff] = indfuns.centroid_progress(w,f,centroids,um_per_pix)
            ratio,product = indfuns.PCA_metrics(w,f,centroids,um_per_pix=um_per_pix)
            indicator_vals[10][w][ff] = ratio
            indicator_vals[11][w][ff] = product
            
            # calculate indicators based on two frames
            if ff >= 1:
                centerline0 = indfuns.smooth_centerline(np.float64(centerlines[w][ff-1]))
            
                indicator_vals[0][w][ff] = indfuns.ends_mov_bias(centerline0,centerline1,um_per_pix)
                indicator_vals[1][w][ff] = indfuns.out_of_track_centerline_mov(centerline0,centerline1,um_per_pix)
                lat, lon = indfuns.lat_long_movement(centerline0,centerline1,um_per_pix)
                indicator_vals[4][w][ff] = lat
                indicator_vals[5][w][ff] = lon
                indicator_vals[7][w][ff] = indfuns.angular_sweep(centerline0,centerline1,supp = True)
            
            ff = ff + 1


# save indicator values
save_name = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20211129_indicator_vals_TEST.p'
pickle.dump(indicator_vals, open(save_name, "wb" ) )