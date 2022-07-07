# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:50:32 2022


This module contains functions that calculate features used to detect nictation
based on tracking and video data. Most features require tracking information
from one or two frames, or in one case the raw video, to calculate. A few
require <num_f> frames in the past or future, and return NaN otherwise. These
are currently ambiguous to the head / tail orientation of the worm.


This is based on the earlier indicator_functions.py


Issues and improvements:
    
    -diff_img_act has hard-coded defaults like the radius around the 
     centerline to use in masking the bw activity image, as well as the 
     smoothing and thresholding parameters.
    
    -at least one feature is sometimes returning infinity, though rarely, 
     about once every 25,000 worm-frames.
     
    -it might be good to have a change in intensity and width feature as the
     worm is narrower-looking when it is nictating.
     
    -area might be a useful feature, but would have to be recorded during
     tracking and probably normalized against a population median.


@author: PDMcClanahan
"""


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
import pickle
import time
import sys



def ends_mov_bias(centerline0, centerline1, um_per_pix):
    '''ends movement bias - The absolute value of the difference between the
    the distance travelled by the head and the distance travelled by the tail.
    Returns the same value regardless of orientation'''
    
    try:
        
        d1 = np.linalg.norm(centerline1[0]-centerline0[0])
        d2 = np.linalg.norm(centerline1[1]-centerline0[1])
        return abs(d1-d2) * um_per_pix
    
    except:
        
        return np.nan



def head_tail_mov_bias(centerline0, centerline1, um_per_pix):
    '''head to tail movement bias - The signed value of the difference between
    the distance travelled by the head and the distance travelled by the tail.
    Requires centerlines to be oriented head to tail'''
    
    try:
    
        dh = np.linalg.norm(centerline1[0]-centerline0[0])
        dt = np.linalg.norm(centerline1[1]-centerline0[1])
        return (dh-dt) * um_per_pix
    
    except:
        
        return np.nan



def out_of_track_centerline_mov(centerline0,centerline1,um_per_pix):
    '''out of track centerline movement - The sum of the minimum distance of
    each point in the current centerline to *any* point in the previous 
    centerline'''
    
    try:
    
        offsets = np.empty(len(centerline0))
        for i in range(len(offsets)):
            distances = np.empty(len(centerline0))
            for j in range(len(distances)):
                distances[j] = np.linalg.norm(centerline1[i]-centerline0[j])
            offsets[i] = np.min(distances)
        return sum(offsets)/len(offsets) * um_per_pix
    
    except:
        
        return np.nan



def blur(vid, vid_f, w, f, centroid, centerline, um_per_pix, halfwidth):
    '''# blurriness - Variance of the laplacian of the cropped image with only
    the area near the centerline shown.  Reference: 
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/'''
    
    blur_radius = int(np.round(75*(1/um_per_pix))) # pixels
    vid.set(cv2.CAP_PROP_POS_FRAMES, vid_f)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = np.zeros(np.shape(img))
    for p in range(len(centerline)):
        try:
            bw[np.round(centerline[p][1]).astype(np.uint16),
               np.round(centerline[p][0]).astype(np.uint16)] = 255
        except:
            pass
    
    kernel = np.ones((blur_radius,blur_radius),np.uint8)
    bw = cv2.dilate(bw,kernel,iterations = 1)
    img[np.where(bw==0)]=0
    
    canvas = np.uint8(np.zeros((np.shape(img)[0]+halfwidth*2,
                                np.shape(img)[1]+halfwidth*2)))
    canvas[halfwidth:np.shape(img)[0]+halfwidth,
           halfwidth:np.shape(img)[1]+halfwidth] = img
    centroid = np.uint16(np.round(centroid))
    crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),
                  centroid[0]:(2*halfwidth+centroid[0])]
    
    return cv2.Laplacian(crop, cv2.CV_64F).var()



def bkgnd_sub_blur(vid, vid_f, w, f, centroid, centerline, um_per_pix, 
                        halfwidth, background):
    
    '''blurriness - Variance of the laplacian of the cropped background-
    subtracted image with only the area near the centerline shown.  Reference: 
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/'''
    
    
    blur_radius = int(np.round(75*(1/um_per_pix))) # pixels
    vid.set(cv2.CAP_PROP_POS_FRAMES, vid_f)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.absdiff(img,background)
    
    bw = np.zeros(np.shape(img))
    for p in range(len(centerline)):
        try:
            bw[np.round(centerline[p][1]).astype(np.uint16),
               np.round(centerline[p][0]).astype(np.uint16)] = 255
        except:
            pass
    
    kernel = np.ones((blur_radius,blur_radius),np.uint8)
    bw = cv2.dilate(bw,kernel,iterations = 1)
    img[np.where(bw==0)]=0
    
    canvas = np.uint8(np.zeros((np.shape(img)[0]+halfwidth*2,
                                np.shape(img)[1]+halfwidth*2)))
    canvas[halfwidth:np.shape(img)[0]+halfwidth,
           halfwidth:np.shape(img)[1]+halfwidth] = img
    centroid = np.uint16(np.round(centroid))
    crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),
                  centroid[0]:(2*halfwidth+centroid[0])]
    
    return cv2.Laplacian(crop, cv2.CV_64F).var()



def bkgnd_sub_ends_blur_diff(vid, vid_f, w, f, centerline, um_per_pix, 
                        halfwidth, background):
    
    '''# blurriness - Variance of the laplacian of the cropped background-
    subtracted image with only the area near the centerline shown.  Reference: 
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/'''
    
    
    blur_radius = int(np.round(75*(1/um_per_pix))/2) # pixels
    vid.set(cv2.CAP_PROP_POS_FRAMES, vid_f)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.absdiff(img,background)
    
    bw = np.zeros(np.shape(img))
    # for p in range(len(centerline)):
    #     bw[np.round(centerline[p][1]).astype(np.uint16),
    #        np.round(centerline[p][0]).astype(np.uint16)] = 255
    
    # kernel = np.ones((blur_radius,blur_radius),np.uint8)
    # bw = cv2.dilate(bw,kernel,iterations = 1)
    # img[np.where(bw==0)]=0
    
    canvas = np.uint8(np.zeros((np.shape(img)[0]+halfwidth*2,np.shape(img)[1]+
                                halfwidth*2)))
    canvas[halfwidth:np.shape(img)[0]+halfwidth,
           halfwidth:np.shape(img)[1]+halfwidth] = img
    
    crop1 = canvas[halfwidth + np.round(centerline[0][1].astype(np.uint16)) - 
                   blur_radius : halfwidth + 
                   np.round(centerline[0][1].astype(np.uint16)) + blur_radius,
                   halfwidth + np.round(centerline[0][0].astype(np.uint16)) - 
                   blur_radius : halfwidth + 
                   np.round(centerline[0][0].astype(np.uint16)) + blur_radius]
    
    crop2 = canvas[halfwidth + np.round(centerline[-1][1].astype(np.uint16)) -
                   blur_radius : halfwidth + 
                   np.round(centerline[-1][1].astype(np.uint16)) + \
                       blur_radius,
                   halfwidth + np.round(centerline[-1][0].astype(np.uint16)) - 
                   blur_radius : halfwidth + 
                   np.round(centerline[-1][0].astype(np.uint16)) + \
                       blur_radius]
    
    
    return np.abs(cv2.Laplacian(crop2, cv2.CV_64F).var() - 
                  cv2.Laplacian(crop1, cv2.CV_64F).var())



def total_curvature(centerline):
    '''total curvature - Sum of the angles at each segment in the worm 
    centerline'''
    
    xs = centerline[:,0]
    ys = centerline[:,1]
    
    dxs = np.diff(xs,n=1,axis=0)
    dys = np.diff(ys,n=1,axis=0)
    
    # angle relative to right horizontal, defined -pi to pi
    angles = np.arctan2(dys,dxs) 
    dangles = np.abs(np.diff(angles,n=1,axis=0))
    dangles[np.where(dangles>np.pi)] = \
        2 * np.pi-dangles[np.where(dangles > np.pi)]

    return np.sum(dangles)



def lat_long_movement(centerline0, centerline1, um_per_pix):
    '''lateral movement - Average movement of corresponding point in the
    lateral (normal) direction, defined as a right angle from the line 
    connecting the point behind and in front of the point in question'''
    
    if centerline0 is not None:
    
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
    
    else:
        
        return np.nan, np.nan



def centroid_path_length_past(w, f, centroids, um_per_pix, num_f):
    '''centroid path length in the past - Distance travelled by the centroids
    num_f frames in the past to the current frame'''
    
    
    cp = 0
    if f - num_f >= 0:
        #import pdb; pdb.set_trace()
        for ff in range(f-num_f,f,1):
            cp = cp + \
                np.linalg.norm(centroids[ff]-centroids[ff+1]) * um_per_pix
    else:
        cp = np.nan
    return cp



def centroid_path_length_fut(w, f, centroids, um_per_pix, num_f):
    '''centroid path length in the future - Distance travelled by the 
    centroids num_f frames in the past to the current frame'''

    cp = 0
    if f + num_f < len(centroids):
        for ff in range(f,f+num_f,1):
            cp = cp + \
                np.linalg.norm(centroids[ff]-centroids[ff+1]) * um_per_pix
    else:
        cp = np.nan
    
    return cp



def head_tail_path_bias(centerlines,w,f,num_f,um_per_pix):
    '''head / tail path - The absolute value of the difference between the 
    total distance covered by the head minus that covered by the tail from f -
    offset to f + offset'''

    centerlines_w = centerlines[w]
    hp = 0
    tp = 0
    
    if f >= num_f and f + num_f < len(centerlines_w):
        for ff in range(f-num_f,f+num_f,1):
            hp = hp + np.linalg.norm(centerlines_w[ff][0][0] - \
                                     centerlines_w[ff+1][0][0]) * um_per_pix
            
            tp = tp + np.linalg.norm(centerlines_w[ff][0][-1] - \
                                     centerlines_w[ff+1][0][-1]) * um_per_pix
        
        abs_bias = np.abs(hp - tp)
    
    else: 
        abs_bias = np.nan
    
    return abs_bias
# not sure how to use this output yet - measure of convolutedness multiplied
# by scale?



def angular_sweep(centerline0,centerline1,supp = True):
    '''angular sweep - The change in angle from head to tail from centerline0
    to centerline1 (the angle 'swept' by the head during waving)'''
    
    try:
    
        angle_f0 = np.arctan2(centerline0[0,1]-centerline0[-1,1],
                              centerline0[0,0]-centerline0[-1,0])
        angle_f1 = np.arctan2(centerline1[0,1]-centerline1[-1,1],
                              centerline1[0,0]-centerline1[-1,0])
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
            
    except:
        
        return np.nan
    


def body_length(centerline,um_per_pix = 1):
    '''body length - The apparent length of the worm body calculated by adding
    up the lengths of all the centerline segments and multiplying that sum by 
    the scale factor'''
    
    body_length = 0
    for p in range(1,len(centerline)):
        body_length = body_length + np.linalg.norm(
            centerline[p-1]-centerline[p])
    return body_length * um_per_pix



def centroid_progress(w, f, centroids, offset, um_per_pix = 1):
    '''centroid progress - The net distance covered by the centroid from f - 
    offset to f + offset'''
    
    centroids_w = centroids[w]
    
    if f-offset >= 0 and f+offset < len(centroids_w):
        
        metric = np.linalg.norm(centroids_w[f-offset]-centroids_w[f+offset]) \
            * um_per_pix
    
    else:
        metric = np.nan
    
    return metric



# Gaussian shape - requires centroids from x frames in the past and future
from sklearn.decomposition import PCA

def PCA_metrics(w,f,centroids,offset = 5,um_per_pix = 1, show = False):
    '''PC ratio and products - The ratio and product of the principal 
    componants of the centroid positions from f - offset to f + offset; the 
    first is a measure of eccentricity, the second a measure of spread'''
    
    # scale need to be defined to make this imaging-system independent
    if f >= offset and f + offset < len(centroids[w]):
        
        pts = np.array(centroids[w])[f-offset:f+offset]*(um_per_pix)
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
            for length, vector in zip(pca.explained_variance_, 
                                      pca.components_):
                ax.annotate('', pca.mean_, pca.mean_+vector*np.sqrt(length),
                            arrowprops=arrowprops)
            
        ratio = np.sqrt(pca.explained_variance_[0])/ \
            np.sqrt(pca.explained_variance_[1])
        product = np.sqrt(pca.explained_variance_[0])* \
            np.sqrt(pca.explained_variance_[1])
    
    else:
        ratio = np.nan; product= np.nan
        
    return ratio, product



def diff_img_act(vid, centerlines, ffs):
    '''difference image activity - The number of pixels near the worm 
    centerline that change by more than the threshold value from the previous
    to current frame'''

    act_radius = 8

    # list of lists to hold the difference image activity values for each worm
    # add NaN values for the first activity value for worms that appear in the
    # first frame of the video
    act_list = []
    for w in range(len(centerlines)):
        if ffs[w] == 0:
            act_list.append([np.nan])
        else:
            act_list.append([])
    
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,img0 = vid.read(); img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
     # go through the remaining video frames
    for f in range(1,num_f):
        print('Finding difference image activity in frame ' + str(f+1) + \
              ' of ' + str(num_f) +'.')
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        
        # calculate the difference image
        ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           
        diff = cv2.absdiff(img,img0)
        smooth = cv2.GaussianBlur(diff,(21,21),1,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,10,1,cv2.THRESH_BINARY)
        
        # determine which worms are tracked in this frame
        for w in range(len(ffs)):
            if f >= ffs[w] and f < ffs[w] + len(centerlines[w]):
                
                # mask the difference image and find the activity
                centerline = centerlines[w][f-ffs[w]][0]
                mask = np.zeros(np.shape(img))
                for p in range(len(centerline)):
                    # current centerline
                    try:
                        mask[np.round(centerline[p][1]).astype(np.uint16),
                           np.round(centerline[p][0]).astype(np.uint16)] = 255
                    except:
                        pass
                    
                    # previous frame centerline? - not sure if this should be
                    # included as it would be unfair in cases where tracking
                    # just began
                kernel = np.ones((act_radius,act_radius),np.uint8)
                mask = cv2.dilate(mask,kernel,iterations = 1)
                
                # append the activity to the worm's activity list
                act_list[w].append(np.sum(bw[np.where(mask == 255)]))
                    
        img0 = copy.copy(img)
    
    # re-arrange the activity into one large list
    activity = []
    for act_w in act_list:
        activity += act_w
    
    
    return activity

    

