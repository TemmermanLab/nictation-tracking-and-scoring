# -*- coding: utf-8 -*-
"""
Based on eigenworm_functions.py.  I made minor modifications to the functions
to account for changes in how centerlines are saves, flagged, etc.  This
version makes eigenworms with 49 angles / segments (or 50 points if drawn).

Issues:
    -currently centerlines are resampled to 100 angles using a spline fit,
    this is probably overkill if this is to be used in a deformable model

Created on Fri Dec 10 15:28:13 2021
@author: P. D. McClanahan (pdmcclanahan@gmail.com)
"""

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb, traceback, sys, code
from scipy import interpolate
from scipy.interpolate import interp1d
#from PIL import Image as Im, ImageTk, ImageDraw
#import tkinter as tk
#from tkinter import filedialog, Label
#from tkinter import *
import pickle
import time

# EIGENWORM ANALYSIS
def centerlines_to_angles(centerlines):
    # this version is based on Andre Brown's MATLAB code (below). It does not
    # change the result, but runs much faster
    # dX = diff(x,1,2);
    # dY = diff(y,1,2);

    # % calculate tangent angles.  atan2 uses angles from -pi to pi instead...
    # % of atan which uses the range -pi/2 to pi/2.
    # angles = atan2(dY, dX);

    # % need to deal with cases where angle changes discontinuously from -pi
    # % to pi and pi to -pi.  
    # angles = unwrap(angles,[],2);

    
    all_angles = list()
    for w in range(len(centerlines)):
        print('Finding angles in worm '+str(w+1)+' of '+str(len(centerlines)))
        w_angles = list(); counter = 0
        for f in range(np.shape(centerlines[w])[0]-1):
            counter = counter + 1;
            # extract midline pixel coordinates
            x = centerlines[w][f][0][:,1]
            y = centerlines[w][f][0][:,0]
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
            # calculate the angle at each joint and sum them
            xs = new_pts[0]
            ys = new_pts[1]
            
            dxs = np.diff(xs,n=1,axis=0)
            dys = np.diff(ys,n=1,axis=0)
            
            angles = np.arctan2(dys,dxs)
            angles = np.unwrap(angles)
                
            angles = angles - np.mean(angles)
            txt = 'frame '+str(f)
            # if w > 3 and f > 300:
            #     plot_worm_from_angles(angles,txt); time.sleep(0.001)
            w_angles.append(np.float32(angles))
        all_angles.append(w_angles)
    return(all_angles)

def centerlines_to_angles_old(centerlines):
    all_angles = list()
    for w in range(len(centerlines)):
        print('Finding angles in worm '+str(w+1)+' of '+str(len(centerlines)))
        w_angles = list(); counter = 0
        for f in range(np.shape(centerlines[w])[0]-1):
            counter = counter + 1;
            # extract midline pixel coordinates
            x = centerlines[w][f,:,1]
            y = centerlines[w][f,:,0]
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
            # calculate the angle at each joint and sum them
            angles = np.zeros((np.shape(new_pts)[1]-1))
            for p in reversed(range(1,np.shape(new_pts)[1]-1)):
                a = [new_pts[0][p]-new_pts[0][p+1],new_pts[1][p]-new_pts[1][p+1]]
                b = [new_pts[0][p-1]-new_pts[0][p],new_pts[1][p-1]-new_pts[1][p]]
                ang = np.arcsin(np.cross(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                angles[p-1] = angles[p]+ang;
                
            angles = angles - np.mean(angles)
            txt = 'frame '+str(f)
            # if w > 3 and f > 300:
            #     plot_worm_from_angles(angles,txt); time.sleep(0.001)
            w_angles.append(np.float32(angles))
        all_angles.append(w_angles)
    return(all_angles)

def plot_worm_from_points(x,y,txt = ''):
    plt.figure()
    plt.plot(x,y,'k')
    #plt.xlim(-5,105)
    #plt.ylim(-50,50)
    plt.axis('equal')
    plt.show()

def plot_worm_from_angles(angles,txt = ''):
    x = np.empty((np.shape(angles)[0]+1)); x[0] = 0
    y = np.empty((np.shape(angles)[0]+1)); y[0] = 0
    for p in range(np.shape(x)[0]-1):
        # x[p+1] = x[p] + np.cos(angles[p-1])
        # y[p+1] = y[p] + np.sin(angles[p-1])
        x[p+1] = x[p] + np.cos(angles[p])
        y[p+1] = y[p] + np.sin(angles[p])
    x = x - np.mean(x); y = y-np.mean(y);
    plt.figure()
    plt.plot(x,y,'k')
    plt.axis('equal')
    plt.xlim(-50,50)
    plt.ylim(-25,25)
    plt.title(txt)
    plt.show()
    
def get_xy_from_angles(angles):
    x = np.empty((np.shape(angles)[0]+1)); x[0] = 0
    y = np.empty((np.shape(angles)[0]+1)); y[0] = 0
    for p in range(np.shape(x)[0]-1):
        # x[p+1] = x[p] + np.cos(angles[p-1])
        # y[p+1] = y[p] + np.sin(angles[p-1])
        x[p+1] = x[p] + np.cos(angles[p])
        y[p+1] = y[p] + np.sin(angles[p])
    x = x - np.mean(x); y = y-np.mean(y);
    return x,y

def angles_to_eigenstuff(all_angles):
    # takes the angles decribing worm centerline postures and creates a
    # covariance matrix and
    for w in range(np.shape(all_angles)[0]):
        print('Arranging data in worm '+str(w))
        for f in range(np.shape(all_angles[w])[0]):
            col = np.empty((np.shape(all_angles[0])[1],1))
            col[:,0] = all_angles[w][f]
            if w == 0 and f == 0:
                data = np.reshape(copy.copy(col),(np.shape(all_angles[0])[1],1))
            else:
                data = np.hstack((data,copy.copy(col)))
    
    M = np.cov(data)
    EVal,EVec = np.linalg.eig(M)
    return(M,EVal,EVec)

def show_covar_matrix(M, txt = "Postural Covariance Matrix"):
    fig, ax = plt.subplots()
    im = ax.imshow(M,cmap = 'jet')
    matplotlib.colors.Normalize()
    fig.colorbar(im)
    ax.invert_yaxis()
    # ax.axis('equal')
    ax.set_xlabel('Body coordinate')
    ax.set_xticks((-.5,np.shape(M)[0]-0.5))
    ax.set_xticklabels(('0','1'))
    ax.set_ylabel('Body coordinate')
    ax.set_yticks((-0.5,np.shape(M)[0]-0.5))
    ax.set_yticklabels(('0','1'))
    ax.set_title(txt)
    fig.tight_layout()
    plt.show()

def plot_top_four_eigenworms_points(EVec , txt = 'First Four Eigenworms'):
    fig, axs = plt.subplots(4)
    fig.suptitle(txt)
    x,y = get_xy_from_angles(EVec[:,0])
    axs[0].plot(x, y)
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    x,y = get_xy_from_angles(EVec[:,1])
    axs[1].plot(x, y)
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    x,y = get_xy_from_angles(EVec[:,2])
    axs[2].plot(x, y)
    axs[2].set_xticks(())
    axs[2].set_yticks(())
    x,y = get_xy_from_angles(EVec[:,3])
    axs[3].plot(x, y)
    axs[3].set_xticks((-np.shape(EVec)[0]/2.0,np.shape(EVec)[0]/2.0))
    axs[3].set_xticklabels(('0','1'))
    axs[3].set_xlabel('Body coordinate')
    axs[3].set_yticks(())
    
def plot_top_four_eigenworms(EVec , txt = 'First Four Eigenworms'):
    fig, axs = plt.subplots(4,figsize = (2.5,4.5))
    fig.suptitle(txt)
    #fig.add_subplot(111, frame_on=False)
    #plt.tick_params(labelcolor="none", bottom=False, left=False)
    #plt.ylabel('theta')
    axs[0].plot(EVec[:,0])
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    axs[1].plot(EVec[:,1])
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    axs[2].plot(EVec[:,2])
    axs[2].set_xticks(())
    axs[2].set_yticks(())
    axs[3].plot(EVec[:,3])
    axs[3].set_xticks(())
    axs[3].set_yticks(())
    # x,y = get_xy_from_angles(EVec[:,1])
    # axs[1].plot(x, y)
    # axs[1].set_xticks(())
    # axs[1].set_yticks(())
    # x,y = get_xy_from_angles(EVec[:,2])
    # axs[2].plot(x, y)
    # axs[2].set_xticks(())
    # axs[2].set_yticks(())
    # x,y = get_xy_from_angles(EVec[:,3])
    # axs[3].plot(x, y)
    # axs[3].set_xticks((-49,49))
    # axs[3].set_xticklabels(('0','1'))
    axs[3].set_xlabel('Body coordinate')
    axs[3].set_yticks(())
   
def plot_eigenvector_variance(EVal, n = 8):
    # this function plots the % variance accounted for by each eigenworm in a
    # manner similar to Figure 2b in Stephens et al. It assumes that the % var
    # explained = eigenvalue / sum(eigenvalues) and that all the eigenvalues
    # together explain all the variance (I am not sure if this is true, there
    # may be additional unexplained variance)
    to_plot = np.empty(n)
    for ev in range(n):
        to_plot[ev] = np.sum(EVal[0:ev+1])/np.sum(EVal)
    
    plt.plot(np.linspace(1,n,n),to_plot,'r.',markersize=15,markeredgecolor='k')
    plt.plot((0.4,n+0.4),(1,1),'k--',linewidth = 1.5)
    plt.xlabel('K'); plt.ylabel('variance captured')
    plt.yticks((0,1))
    plt.ylim((0,1.05))
    plt.xlim((0.4,n+0.4))
    plt.title('Variance Captured by First K Eigenworms')
    
def get_eigenworm_coefficients(angles, EVecs, n):
    A = EVecs[:,0:n]
    x = np.linalg.lstsq(A,angles,rcond=None)
    coeffs = x[0]
    reconst = np.sum(x[0][i]*EVecs[:,i] for i in range(n))
    return coeffs, reconst

def resample_angles(all_angles,N):
    '''Uses spline fitting to resample the centerline angles such that the
    resulting worm will have N points, or N-1 angles / segments'''
    #import pdb; pdb.set_trace()
    all_angles_new = copy.deepcopy(all_angles)
    for w in range(len(all_angles)):
        for f in range(len(all_angles[w])):
            angles_orig = copy.deepcopy(all_angles[w][f])
            fun = interp1d(np.linspace(0,100,len(angles_orig)), angles_orig, kind='cubic')
            x_new = np.linspace(0, 100, N-1)
            angles_new = fun(x_new)
            all_angles_new[w][f] = angles_new       
            
    print('boo!!!!?')
    return all_angles_new



try:
    # load centerlines
    # import sys
    # sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2')
    # import tracker_classes as tracker
    # tracker1 = tracker.Tracker(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vids_cropped\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi')
    # tracker1.load_centerlines()
    # tracker2 = tracker.Tracker(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vids_cropped\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3.avi')
    # tracker2.load_centerlines()
    # centerlines1 = tracker1.centerlines
    # flags1 = tracker1.centerline_flags
    # centerlines2 = tracker2.centerlines
    # flags2 = tracker2.centerline_flags
    # centerlines = centerlines1+centerlines2
    # flags = flags1+flags2
    
    
    # # clean up centerlines
    # for w in range(len(centerlines)):
    #     for f in range(len(centerlines[w])-1,-1,-1):
    #         if flags[w][f]:
    #             centerlines[w].pop(f)
    
        
    # # convert to angles
    # angles = centerlines_to_angles(centerlines)
    
    
    # # save angles
    # filename = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\20211212_angles.p'
    # with open(filename, 'wb') as f:
    #     pickle.dump(angles, f)
    
    # print('Now restart kernel and run again to reload angles and finish calculating eigenworms')
    
    # load
    N = 50 # number of *points* (number of segments is N-1)
    filename = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20211212_angles.p'
    with open(filename, 'rb') as f:
        angles = pickle.load(f)
        
    angles2 = resample_angles(angles,N)
    plt.plot(angles[0][0],'b.')
    plt.plot(np.linspace(0,len(angles[0][0])-1,len(angles2[0][0])),angles2[0][0],'r.')
    plt.show()
    angles = copy.deepcopy(angles2)
    
    M,EVal,EVec = angles_to_eigenstuff(angles)
    plot_top_four_eigenworms(EVec)
    plot_top_four_eigenworms_points(EVec)
    show_covar_matrix(M)
    
    # save eigenworms
    eigendict = {
        'matrix' : M,
        'eigenvalues' : EVal,
        'eigenvectors' : EVec}
    
    filename = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\nictation\test_files\20211212_Sc_eigenworms_50b.p'
    filename = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20211212_Sc_eigenworms_50b.p'
    with open(filename, 'wb') as f:
        pickle.dump(eigendict, f)
    
    # load eigenworms
    filename = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20211212_Sc_eigenworms_50b.p'
    with open(filename, 'rb') as f:
        eigendict2 = pickle.load(f)
    M2 = eigendict2['matrix']
    EVal2 = eigendict2['eigenvalues']
    EVec2 = eigendict2['eigenvectors']
    
    pass
    
except:
    
    import pdb
    import sys
    import traceback
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)