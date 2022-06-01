# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:19:30 2022

This module contains methods for generating plots and spreadsheets summarizing
the tracking and scoring.

Issues and improvements:
    -plot of detected vs tracked worms in each frame (to be called during
     tracking), also inference time, number centerlines flagged, and fixing
     time
    -traces of every track in a video colored by behavior
    -
    

@author: Temmerman Lab
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.split(__file__)[0])
import tracker_classes as trkr


def tracks_vs_time_plot(vid_file, centroids_file):
    '''Creates a Gantt plot of all the worm tracks in <centroid_file> in the
    video <vid_file> and saves it in a subfolder "plots" alongside the
    centroids file'''
    
    vid = cv2.VideoCapture(vid_file)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    centroids, ffs = trkr.Tracker.load_centroids_csv(centroids_file)

    canvas = 255 * np.ones((len(centroids),num_f))
    for w in range(len(centroids)):
        canvas[w,ffs[w]:ffs[w]+len(centroids[w])] = 0
    
    
    num_trks = len(ffs)
    tot_h_trks = (len(np.where(canvas == 0)[0])/fps)/3600
    s_per_trk = (len(np.where(canvas == 0)[0])/fps) / num_trks
    vid_name = os.path.split(vid_file)[1]
    
    
    plt.imshow(canvas, cmap = 'gray', aspect = 10)
    plt.ylabel('track')
    plt.xlabel('frame')
    plt.title('Tracks in ' + vid_name + '\n' + str(num_trks) + ' trks, ' + \
              str(round(tot_h_trks,2)) + ' h total, min 10 s / trk' )
        
    save_path = os.path.split(centroids_file)[0]+'\\plots'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '\\tracks_vs_time', dpi = 200)



if __name__ == '__main__':
    
    try:
        
        vf = 'C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictati' + \
            'on_dataset\\Ce_R2_d21.avi'
        
        cf = 'C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictati' + \
            'on_dataset\\Ce_R2_d21_tracking\\centroids.csv'
        
        tracks_vs_time_plot(vf, cf)
        
    except:
        
        import pdb
        import sys
        import traceback
        
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
    
    
    