# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:55:50 2021

Custom quantitative measures imagined to be correlated with nictation

Measures require one or two frames to calculate.

@author: PDMcClanahan
"""

# modules
import cv2
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb, traceback, sys, code
from scipy import interpolate
from PIL import Image as Im, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import *
import pickle
import time
import sys
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\scripts\nictation\tracking')
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\scripts\nictation\eigenworms')
import other_functions as track_f
# import eigenworm_functions as eigen_f

testing = False

def smooth_centerline(centerline):
    x = centerline[:,0]
    y = centerline[:,1]
    
    # delete redundant points
    for p in reversed(range(np.shape(x)[0]-1)):
        if x[p] == x[p+1] and y[p] == y[p+1]:
            x = np.delete(x,p+1,0); y = np.delete(y,p+1,0)
    
    # fit to a spline to make smooth and continuous
    X = (x,y)
    tck,u = interpolate.splprep(X,s=20) # s may need adjusting to avoid
    # over- or underfitting
    unew = np.arange(0,1,.01)
    new_pts = interpolate.splev(unew, tck)
    smooth_centerline = np.swapaxes(np.array([new_pts[0],new_pts[1]]),1,0)
    
    return smooth_centerline


# # load test inputs
# centroids = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centroids_clean.p','rb'))
# centerlines = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centerlines_clean.p','rb'))
# scores = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\052021_trinary_scoring\manual_nictation_scores.p','rb'))
# vid = cv2.VideoCapture(r'E:\20210212_Cu_ring_test\dauers 14-14-56.avi')
# bkgnd = track_f.get_background(vid)

# w = 0
# f = 1

# centerline0 = smooth_centerline(np.float64(centerlines[w][f-1]))
# centerline1 = smooth_centerline(np.float64(centerlines[w][f]))
# # plt.plot(centerline0[:,0],centerline0[:,1])
# # plt.plot(centerline1[:,0],centerline1[:,1])
# # plt.plot(centerline1[0,0],centerline1[0,1],'k.')


# um_per_pix = 3.5

# k_sz = (25,25)
# k_sig = 2.0
# bw_thr = 20

# halfwidth = 60

# # params = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\tracking_params.p','rb'))


##############################################################################
# metric functions


# end to end movement bias - The absolute value of the difference between the
# distance travelled by one and of the worm and the distance travelled by the
# other. This version does not distinguish the head and tail.

def ends_mov_bias(centerline0,centerline1,um_per_pix):
    d1 = np.linalg.norm(centerline1[0]-centerline0[0])
    d2 = np.linalg.norm(centerline1[1]-centerline0[1])
    return abs(d1-d2) * um_per_pix




# head to tail movement bias - The signed value of the difference between the
# distance travelled by head and of the worm and the distance travelled by the
# tail. Requires accurate H/T discrimination

def head_tail_mov_bias(centerline0,centerline1,um_per_pix):
    dh = np.linalg.norm(centerline1[0]-centerline0[0])
    dt = np.linalg.norm(centerline1[1]-centerline0[1])
    return (dh-dt) * um_per_pix




# out of track centerline movement - The sum of the distances of each point in
# the current centerline from any point in the previous centerline

def out_of_track_centerline_mov(centerline0,centerline1,um_per_pix):
    offsets = np.empty(len(centerline0))
    for i in range(len(offsets)):
        distances = np.empty(len(centerline0))
        for j in range(len(distances)):
            distances[j] = np.linalg.norm(centerline1[i]-centerline0[j])
        offsets[i] = np.min(distances)
    return sum(offsets)/len(offsets) * um_per_pix




# blurriness - variance of the laplacian of the cropped image with only the
# area inside a bw object shown (currently any bw object, not just the worm)
# reference: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

def blur(vid,w,f,centroid1,um_per_pix,bw_thr,k_sz,k_sig,bkgnd,halfwidth):
    blur_radius = np.round(50*(1/um_per_pix))
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(img,bkgnd)
    smooth = cv2.GaussianBlur(diff,tuple(k_sz),k_sig,cv2.BORDER_REPLICATE)
    thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    bw = cv2.dilate(bw,kernel,iterations = 1)
    filtered = copy.copy(diff)
    diff[np.where(bw==0)]=0
    
    canvas = np.uint8(np.zeros((np.shape(diff)[0]+halfwidth*2,np.shape(diff)[1]+halfwidth*2)))
    canvas[halfwidth:np.shape(diff)[0]+halfwidth,halfwidth:np.shape(diff)[1]+halfwidth] = diff
    centroid = np.uint16(np.round(centroid1))
    crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
    
    return cv2.Laplacian(crop, cv2.CV_64F).var()



# total curvature - sum of the angles at each segment in the worm centerline

def total_curvature(centerline):
    xs = centerline[:,0]
    ys = centerline[:,1]
    
    dxs = np.diff(xs,n=1,axis=0)
    dys = np.diff(ys,n=1,axis=0)
    
    angles = np.arctan2(dys,dxs) # angle relative to right horizontal, defined -pi to pi
    dangles = np.abs(np.diff(angles,n=1,axis=0))
    dangles[np.where(dangles>np.pi)] = 2*np.pi-dangles[np.where(dangles>np.pi)]

    return np.sum(dangles)



# lateral movement - movement of corresponding point in the lateral (normal)
# direction, defined as a right angle from the line connecting the point
# behind and in front of the point in question.

def lat_long_movement(centerline0,centerline1,um_per_pix = 1):
    
    lat_dist = []
    long_dist = []
    
    # proceed from tail to head
    for p in range(len(centerline1)-1,0,-1):
           
        # extract needed points, p-1 is closer to the *head*, since we are
        # thinking about this from tail to head, point 0 is one segment closer
        # to the tail
        p0_0 = centerline0[p] # frame 0, point 0
        p0_1 = centerline0[p-1] # frame 0, point 1 
        p1_0 = centerline1[p] # frame 1, point 0 (not actually used)
        p1_1 = centerline1[p-1] # frame 1, point 1
    
        # find the unit vector along the worm centerline in frame 0
        cl_vect_raw = np.array([p0_1[0]-p0_0[0],p0_1[1]-p0_0[1]])
        cl_vect_unit = cl_vect_raw / np.linalg.norm(cl_vect_raw)
        
        # find the vector connecting the point in centerline0 (frame 0) to the
        # same point in centerline1 (frame 1)
        # mvmnt_vect = np.array([p1_1[0]-p0_1[0],p1_1[1]-p0_1[1]])
        mvmnt_vect = p1_1-p0_1
        
        # absolute vals taken below because we do not care about fwd vs
        # backward or left vs right movement
        
        # take the cross product to find the lateral distance
        lat_dist.append(np.absolute(np.cross(mvmnt_vect,cl_vect_unit)))
        
        # take the dot product to find the longitudinal distance
        long_dist.append(np.absolute(np.dot(mvmnt_vect,cl_vect_unit)))
            
    # return the means, argument could be made for returning max lateral, or
    # forward part of lateral if head / tail classification is accurate
    return np.mean(lat_dist) * um_per_pix, np.mean(long_dist) * um_per_pix

# plt.plot(centerline0[:,0],centerline0[:,1])
# plt.plot(centerline1[:,0],centerline1[:,1])
# plt.plot(centerline1[0,0],centerline1[0,1],'k.')

# plt.plot(centerline0_rot[:,0],centerline0_rot[:,1])
# plt.plot(centerline1_rot[:,0],centerline1_rot[:,1])
# plt.plot(centerline1_rot[0,0],centerline1_rot[0,1],'r.')


# centroid progress - requires centroids from x frames in the past and future
def centroid_path(w,f,centroids,um_per_pix = 1):
    centroids_w = centroids[w]
    offset = 5
    metric = 0
    if f-offset > 0 and f+offset+1 < len(centroids_w) and ~np.isnan(centroids_w[f-offset,0]) and ~np.isnan(centroids_w[f+offset+1,0]):
        for ff in range(f-offset,f+offset,1):
            metric = metric + np.linalg.norm(centroids_w[ff]-centroids_w[ff+1]) * um_per_pix
    else:
        metric = np.nan
    return metric


# path of head relative to tail - requires centerlines x frames in the past
# and future

def head_path(centerlines,w,f,offset):
    centerlines_w = centerlines[w]
    head_path = []
    tail_path = []
    if np.shape(centerlines_w)[0] > 1+2*poffset:
        for f in range(offset,np.shape(centerlines_w)[0]-offset):
            head_path = centerlines_w[f-offset:f+offset,0,:]
            tail_path = centerlines_w[f-offset:f+offset,-1,:]
    else: 
        head_path = nan; tail_path = nan
    return head_path, tail_path
# not sure how to use this output yet - measure of convolutedness multiplied
# by scale?


        

# frame to frame angular sweep -  the change in angle from tail to head from
# frame to frame

def angular_sweep(centerline0,centerline1,supp = True):
    
    angle_f0 = np.arctan2(centerline0[0,1]-centerline0[-1,1],centerline0[0,0]-centerline0[-1,0])
    angle_f1 = np.arctan2(centerline1[0,1]-centerline1[-1,1],centerline1[0,0]-centerline1[-1,0])
    angle_sweep = np.abs(angle_f1-angle_f0)
    
    # supplementary return (for if head/tail is frequently flipped)
    if supp:
        if angle_sweep <= np.pi:
            angle = angle_sweep
        else:
            angle =  2*np.pi-angle_sweep
        
        if angle <= np.pi/2:
            return angle
        else:
            angle_supp = np.pi-angle
            return angle_supp
        
    else:
        # non-supplementary return
        if angle_sweep <= np.pi:
            return angle_sweep
        else:
            return 2*np.pi-angle_sweep
    
    



# body length distance from head to tail adding up all the segments
def body_length(centerline1,um_per_pix = 1):
    body_length = 0
    for p in range(1,len(centerline1)):
        body_length = body_length + np.linalg.norm(centerline1[p-1]-centerline1[p])
    return body_length * um_per_pix




# centroid progress - requires centroids from x frames in the past and future
def centroid_progress(w,f,centroids,um_per_pix = 1):
    centroids_w = centroids[w]
    offset = 5
    if f-offset > 0 and f+offset < len(centroids_w) and ~np.isnan(centroids_w[f-offset,0]) and ~np.isnan(centroids_w[f+offset,0]):
        metric = np.linalg.norm(centroids_w[f-offset]-centroids_w[f+offset]) * um_per_pix
    else:
        metric = np.nan
    return metric



# Gaussian shape - requires centroids from x frames in the past and future
from sklearn.decomposition import PCA
def PCA_metrics(w,f,centroids,offset = 5,um_per_pix=1, show = False):
    # scale need to be defined to make this imaging-system independent
    if f >= offset and not np.isnan(centroids[w][f+offset][0]) and not np.isnan(centroids[w][f-offset][0]):
        pts = centroids[w][f-offset:f+offset]*(um_per_pix)
        pca = PCA(n_components=2)
        pca.fit(pts)
        
        # show the points
        if show:
            plt.scatter(pts[:,0],pts[:,1])
            plt.arrow(pca.mean_[0], pca.mean_[1],
                pca.components_[0][0]*2*np.sqrt(pca.explained_variance_[0]),
                pca.components_[0][1]*2*np.sqrt(pca.explained_variance_[0]),
                width = .03, color = 'k')
            plt.arrow(pca.mean_[0], pca.mean_[1],
                pca.components_[1][0]*2*np.sqrt(pca.explained_variance_[1]),
                pca.components_[1][1]*2*np.sqrt(pca.explained_variance_[1]),
                width = .03, color = 'k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('PCs of Centroid Positions')
            plt.axis('equal')
            
            ax = plt.gca()
            arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
            for length, vector in zip(pca.explained_variance_, pca.components_):
                ax.annotate('', pca.mean_, pca.mean_+vector*np.sqrt(length), arrowprops=arrowprops)
            
        ratio = np.sqrt(pca.explained_variance_[0])/np.sqrt(pca.explained_variance_[1])
        product = np.sqrt(pca.explained_variance_[0])*np.sqrt(pca.explained_variance_[1])
    else:
        ratio = np.nan; product= np.nan
        
    return ratio, product
##############################################################################
# script for calculating these indicators for a gold standard dataset

# from sys import getsizeof
# a = 1.0
# getsizeof(centroids) # seems suspect

if testing:

    # load test inputs
    # params = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\tracking_params.p','rb'))
    centroids = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centroids_clean.p','rb'))
    centerlines = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\centerlines_clean.p','rb'))
    scores = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\052021_trinary_scoring\manual_nictation_scores.p','rb'))
    vid = cv2.VideoCapture(r'E:\20210212_Cu_ring_test\dauers 14-14-56.avi')
    bkgnd = track_f.get_background(vid)
    
    um_per_pix = 3.5
    k_sz = (25,25)
    k_sig = 2.0
    bw_thr = 20
    halfwidth = 60
    
    w = 0
    f = 1
    
    metrics = ['ends_mov_bias','head_tail_mov_bias','out_of_track_centerline_mov',
               'blur','total_curvature','lateral_movement','centroid_path',
               'angular_sweep','body_length','centroid_progress']
    
    offset = 5
    
    metric_vals = []
    for m in range(len(metrics)):
        empty_metrics = []
        for w in range(len(scores)):
            empty_scores = np.empty(len(scores[w])+1)
            empty_scores[:] = np.NaN
            empty_metrics.append(empty_scores)
        metric_vals.append(empty_metrics)
    del empty_scores,m,empty_metrics
    
    
    for w in range(len(centroids)):
        ff = 0 # frame number for scores, etc
        for f in range(len(centroids[w])):
            
            if ~np.isnan(centroids[w,f,0]):
                print('Computing custom nictation indicators for worm ',str(w),', frame ',str(f))
                centerline1 = smooth_centerline(np.float64(centerlines[w][ff]))
                centroid1 = centroids[w,f]
                
                metric_vals[3][w][ff] = blur(vid,w,f,centroid1,um_per_pix,bw_thr,k_sz,k_sig,bkgnd)
                metric_vals[4][w][ff] = total_curvature(centerline1)
                metric_vals[6][w][ff] = centroid_path(w,f,centroids)
                metric_vals[8][w][ff] = body_length(centerline1,um_per_pix)
                metric_vals[9][w][ff] = centroid_progress(w,f,centroids)
                
                if ff >= 1:
                    centerline0 = smooth_centerline(np.float64(centerlines[w][ff-1]))
                
                    metric_vals[0][w][ff] = ends_mov_bias(centerline0,centerline1,um_per_pix)
                    metric_vals[1][w][ff] = head_tail_mov_bias(centerline0,centerline1,um_per_pix)
                    metric_vals[2][w][ff] = out_of_track_centerline_mov(centerline0,centerline1,um_per_pix)
                    metric_vals[5][w][ff] = lateral_movement(centerline0,centerline1)
                    metric_vals[7][w][ff] = angular_sweep(centerline0,centerline1)
                
                ff = ff + 1
    
            
    
    
    
    # plt.plot(centerline0[:,0],centerline0[:,1])
    # plt.plot(centerline1[:,0],centerline1[:,1])
    # plt.plot(centerline1[0,0],centerline1[0,1],'k.')
    
    # save metric scores
    save_name = r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\051821_quaternary_scoring\20210524_custom_metric_scores.p'
    pickle.dump(metric_vals, open(save_name, "wb" ) )
    
    
    
    # plot
    categories = ['recumbent','actively nictating','quiescently nictating']
    for m in range(len(metrics)):
        scores_by_category = [[],[],[]]
        for w in range(len(scores)):
            for f in range(len(scores[w])):
                scores_by_category[scores[w][f]].append(metric_vals[m][w][f])
        
        plt.title(metrics[m]+' by Nictation State')
        plt.ylabel('Value')
        plt.xticks(np.linspace(1,len(categories),len(categories)),labels = categories)
        for mm in range(len(categories)):
            xvals = 1+mm+np.linspace(-.4,.4,len(scores_by_category[mm]))
            yvals = scores_by_category[mm]
            plt.plot(xvals,yvals,'.',markersize=3)
        plt.show()
    



