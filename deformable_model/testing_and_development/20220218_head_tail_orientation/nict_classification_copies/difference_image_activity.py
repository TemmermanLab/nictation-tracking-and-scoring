# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:51:23 2021

@author: PDMcClanahan
"""

# import modules
import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import copy
import sys
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\scripts\tracking_main')
import tracking_functions as track_f


# load tracking parameters
param_file = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\tracking_params.p'
params = pickle.load(open(param_file,'rb'))


# load centroids and scores
centroids_file = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centroids_clean.p'
centroids = pickle.load(open(centroids_file,'rb'))
scores_file = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\052021_trinary_scoring\manual_nictation_scores.p'
scores = pickle.load(open(scores_file,'rb'))


# find halfwidth
centerlines_file = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centerlines_clean.p'
centerlines = pickle.load(open(centerlines_file,'rb'))
extents = np.empty(0)
for w in range(len(centerlines)):
    for f in range(np.shape(centerlines[w])[0]):
        extent = np.linalg.norm(np.float32(centerlines[w][f,0,:])-np.float32(centerlines[w][f,-1,:]))
        extents = np.append(extents,extent)
halfwidth = int(np.percentile(extents, 99)/2)
del centerlines


# load video
vid_file = params['vid_path']+'\\'+params['vid_name']
vid = cv2.VideoCapture(vid_file)
bkgnd = track_f.get_background(vid)


# calculate difference image activity
numf = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
activity = list()
for w in range(np.shape(centroids)[0]): activity.append([])
for f in range(numf):
    if f == 0:
        vid.set(cv2.CAP_PROP_POS_FRAMES,f)
        ret,img0 = vid.read()
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    else:
        # load frame
        ret,img1 = vid.read(); img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        print('Finding difference image activity in frame ',str(f),' of ',str(numf))
        
        # calculate difference image
        diff = cv2.absdiff(img1,img0)
        smooth = cv2.GaussianBlur(diff,(21,21),1,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,10,1,cv2.THRESH_BINARY)
        
        # find activity in a box around each centroid
        canvas = np.uint8(np.zeros((np.shape(bw)[0]+halfwidth*2,np.shape(bw)[1]+halfwidth*2)))
        canvas[halfwidth:np.shape(bw)[0]+halfwidth,halfwidth:np.shape(bw)[1]+halfwidth] = bw
        for w in range(np.shape(centroids)[0]):
            if ~np.isnan(centroids[w,f,0]):
                centroid = np.uint16(np.round(centroids[w,f]))
                crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
                activity[w].append(np.sum(crop))
        
        img0 = copy.copy(img1)
    
save_name = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\20210511_diff_img_activity.p'
pickle.dump(activity, open(save_name, "wb" ) )


categories = ['recumbent','actively nictating','quiescently nictating']
scores_by_category = [[],[],[]]
for w in range(len(scores)):
    for f in range(len(scores[w])):
        scores_by_category[scores[w][f]].append(activity[w][f])

plt.title('Activity by Nictation State')
plt.ylabel('Value')
plt.xticks(np.linspace(1,len(categories),len(categories)),labels = categories)
for mm in range(len(categories)):
    xvals = 1+mm+np.linspace(-.4,.4,len(scores_by_category[mm]))
    yvals = scores_by_category[mm]
    plt.plot(xvals,yvals,'.',markersize=3)
plt.show()



