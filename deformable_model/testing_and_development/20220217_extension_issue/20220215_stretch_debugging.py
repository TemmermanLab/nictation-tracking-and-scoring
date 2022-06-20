# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:53:45 2022

@author: Temmerman Lab
"""

import numpy as np
from PIL import Image, ImageTk
import os
import cv2
import copy
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import sys
sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation')

import parameter_GUI
import tracker_classes as tracker
import deformable_worm_module as def_worm


os.environ['KMP_DUPLICATE_LIB_OK']='True'



# set up tracker with a breakpoint before the deformable model call set for
# worm 7


# if w == 7:
#     import pdb; pdb.set_trace()
# deformable_model = def_worm.Eigenworm_model()
        

# # (1) initialize a deformable model            
# vid_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_cropped\super_crop.avi'
# um_per_pix = 1.918
# f = 4

# t = tracker.Tracker(vid_file, um_per_pix)
# t.parameters['bkgnd_meth'] = 'max_merge'
# t.parameters['bkgnd_nframes'] = 10
# t.parameters['k_sig'] = 1.5
# t.parameters['bw_thr'] = 100
# t.parameters['d_thr'] = 150
# t.parameters['area_bnds'] = (2000,6500)
# t.parameters['um_per_pix'] = 1.918
# t.get_background()
# t.track()


# # (2) when the debug point it reached, save the target image and moving centerline
# init_cond_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20220215_ext_debugging_start.p'
# with open(init_cond_file, 'wb') as f:
#     pickle.dump([moving_centerline_shifted,target_image], f)
    
    
# (3) load the example target image and moving centerline, initialize the
# deformable model and run.  Playing with debug points in the extension part 
# of the run method and in centerline_to_parameters shows that that method
# cannot faithfully regenerate the parameters from a centerline that itself
# was generated from parameters.

init_cond_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20220215_ext_debugging_start.p'
with open(init_cond_file, 'rb') as f:
    mov_cline, target_img = pickle.load(f)

deformable_model = def_worm.Eigenworm_model()
deformable_model.set_centerline(mov_cline)
# import matplotlib.pyplot as plt
# plt.imshow(deformable_model.bw_image, cmap = 'gray'); plt.show()

# fit the deformable model to the segmentation by gradient descent
n_iter = 100
lr = [20,20,20,.5,3,3,3,3,3]
grad_step = [1,1,1,1,0.1,0.1,0.1,0.1,0.1]
max_step = [2,2,2,0.02,0.1,0.1,0.1,0.1,0.1]
save_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20220215_extension_debugging'
show = True
vid = False
optimizer = def_worm.Gradient_descent(deformable_model,
                                      target_img,
                                      n_iter, lr,
                                      grad_step, max_step,
                                      save_dir, show, vid)
optimizer.run()


# (4) play around with the centerline_to_parameters method

# def rotate(rot_center, p, angle):
#     """Rotates a point counterclockwise around a point by an angle
#     (degrees)"""
#     angle = np.radians(angle)
#     rcx, rcy = rot_center
#     px, py = p

#     x = rcx + np.cos(angle) * (px - rcx) - np.sin(angle) * (py - rcy)
#     y = rcy + np.sin(angle) * (px - rcx) + np.cos(angle) * (py - rcy)
#     return x, y

# N = 5
# base_length = 253
# n_coeff = 5

# # load eigenworms
# eigenworm_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20211212_Sc_eigenworms_50b.p'
# with open(eigenworm_file, 'rb') as f:
#             eigendict = pickle.load(f)
#             EVecs = eigendict['eigenvectors'][:,0:n_coeff]

# # load a centerline
# parameters = np.array([199.64189211, 200.33110024, 270.62147052,   1.1       ,
#          7.54171994,   3.32793145,  -0.59873046,  -4.53026587,
#         -0.27242353])
# centerline = np.array([[107.39083612, 106.3878604 , 105.65657258, 105.35655172,
#         105.63693042, 106.7110269 , 108.84108073, 112.24248961,
#         116.88658546, 122.24751018, 127.81301377, 133.26606232,
#         138.49443453, 143.58256135, 148.6832512 , 153.88915394,
#         159.25616715, 164.7823699 , 170.34221861, 175.77930401,
#         180.92621694, 185.66871693, 189.92256528, 193.67041411,
#         196.91114687, 199.64189211, 201.8134617 , 203.36940756,
#         204.23630938, 204.37264127, 203.80665141, 202.60060049,
#         200.83047929, 198.52053334, 195.64069884, 192.16304203,
#         188.13249226, 183.64021986, 178.81499274, 173.68812297,
#         168.27399727, 162.70835278, 157.23003544, 151.91329859,
#         146.80488073, 141.97427545, 137.48440123, 133.42143592,
#         130.39413647, 129.25964398],
#        [254.03662972, 248.56174199, 243.04399117, 237.48608298,
#         231.92714928, 226.46576917, 221.32347137, 216.91770542,
#         213.84968819, 212.35274389, 212.2784081 , 213.39403234,
#         215.30308447, 217.55948015, 219.78733092, 221.75683153,
#         223.23179772, 223.89620657, 223.63459928, 222.44360882,
#         220.32473096, 217.41113356, 213.82154721, 209.70645484,
#         205.18119162, 200.33110024, 195.20619619, 189.86209772,
#         184.36402194, 178.79969183, 173.26254351, 167.82877896,
#         162.55175073, 157.48771028, 152.72463257, 148.37880019,
#         144.54017745, 141.25386573, 138.47942179, 136.31249715,
#         135.02114168, 134.9582344 , 135.94230509, 137.5893222 ,
#         139.79939525, 142.56446464, 145.85405207, 149.65834817,
#         154.32908838, 159.77824269]])


# ##################
# # copy of parameters_to_centerline

# def ps_to_cl_orig(params):
#     #import pdb; pdb.set_trace()
#     x_shift = params[0]
#     y_shift = params[1]
#     rot = params[2]
#     stretch = params[3]
#     coeffs = params[4:]
#     N = 50
#     base_length = 253
#     seg_length = stretch*(base_length / N) 
    
#     # calculate centerline angles from eigenworm coefficients
#     angles = np.zeros(N-1)
#     for i in range(len(coeffs)):
#         angles = angles + coeffs[i]*EVecs[:,i]
        
#     # calculate points from angles
#     X = np.empty((np.shape(angles)[0]+1)); X[0] = 0
#     Y = np.empty((np.shape(angles)[0]+1)); Y[0] = 0
#     for p in range(np.shape(X)[0]-1):
#         X[p+1] = X[p] + seg_length * np.cos(angles[p])
#         Y[p+1] = Y[p] + seg_length * np.sin(angles[p])
    
#     # center the centerline on origin
#     # X = X - np.mean(X); Y = Y-np.mean(Y);
#     X = X - X[int(N/2)]; Y = Y - Y[int(N/2)]

#     # rotate points
#     rot_center = [X[int(N/2)],Y[int(N/2)]]
#     for i in range(len(X)):
#         X[i],Y[i] = rotate(rot_center,[X[i],Y[i]],rot)
        
#     # shift points
#     X = X + x_shift
#     Y = Y + y_shift
    
#     cline = np.array([X,Y])
#     return cline

# def cl_to_ps_orig(cline,EVecs):
#     #import pdb; pdb.set_trace()
#     base_length = 253
#     x = cline[1][int(len(cline[0])/2)]
#     y = cline[0][int(len(cline[0])/2)]
    
#     # turn centerline into angles and find the rotation angle            
#     dxs = np.diff(cline[0],n=1,axis=0)
#     dys = np.diff(cline[1],n=1,axis=0)
#     angles = np.degrees(np.unwrap(np.arctan2(dys,dxs)))
#     rot = np.mean(angles) + 360
#     angles = angles - rot
    
#     # turn angles into eigenworm coefficient parameters
#     A = EVecs[:,0:n_coeff]
#     X = np.linalg.lstsq(A,np.radians(angles),rcond=None)
#     coeffs = X[0]
    
#     # find the total length and calculate the stretch coefficient
#     length = np.sum(np.sqrt(np.square(np.diff(cline[1]))+np.square(np.diff(cline[0]))))
#     stretch = length/base_length
    
#     # set parameters
#     params = [x,y,rot,stretch, X[0][0],X[0][1],X[0][2],X[0][3],X[0][4]]
#     return params


# # copy of centerline to parameters
# def cl_to_ps_fixed(cline,EVecs):
#     #import pdb; pdb.set_trace()
#     N = 50

#     base_length = 253
#     x = cline[1][int(len(cline[0])/2)]
#     y = cline[0][int(len(cline[0])/2)]
    
#     # turn centerline into angles and find the rotation angle            
#     dxs = np.diff(cline[0],n=1,axis=0)
#     dys = np.diff(cline[1],n=1,axis=0)
#     angles = np.degrees(np.unwrap(np.arctan2(dys,dxs)))
#     rot = np.mean(angles) + 360
#     angles = angles - np.mean(angles)
    
#     # turn angles into eigenworm coefficient parameters
#     A = EVecs[:,0:n_coeff]
#     X = np.linalg.lstsq(A,np.radians(angles),rcond=None)
#     coeffs = X[0]
    
#     # find the total length and calculate the stretch coefficient
#     length = np.sum(np.sqrt(np.square(np.diff(cline[1]))+np.square(np.diff(cline[0]))))
#     stretch = length/base_length
    
#     # what are the new angles?
#     angles_new = np.zeros(N-1)
#     for i in range(n_coeff):
#         angles_new = angles_new + coeffs[i]*EVecs[:,i]
#     angles_new = np.degrees(angles_new)
#     plt.plot(angles)
#     plt.plot(angles_new)
#     plt.show()
    
#     # set parameters
#     params = [x,y,rot,stretch, X[0][0],X[0][1],X[0][2],X[0][3],X[0][4]]
#     return params

#     # # new order
#     # base_length = 253
#     # # shift centerline
#     # x = cline[1][int(len(cline[0])/2)]
#     # y = cline[0][int(len(cline[0])/2)]
#     # cline[1] = cline[1]-x
#     # cline[0] = cline[0]-y
    
#     # # rotate centerlines
#     # dys = np.diff(cline[0],n=1,axis=0)
#     # dxs = np.diff(cline[1],n=1,axis=0)
#     # angles = np.degrees(np.unwrap(np.arctan2(dxs,dys)))
#     # rot = np.mean(angles) + 360
#     # for p in range(len(cline)):
#     #     cline[:,p] = rotate((0,0), cline[:,p], -rot)
    
#     # # shift so that tail is touching origin
#     # for p in range(len(cline)):
#     #     cline[:,p] = cline[:,p] - cline[:,0]
    
    
#     # # find angles again and calculate eigenworm coefficients
#     # dys = np.diff(cline[0],n=1,axis=0)
#     # dxs = np.diff(cline[1],n=1,axis=0)
#     # angles = np.degrees(np.unwrap(np.arctan2(dxs,dys)))
#     # #rot = np.mean(angles) + 360
#     # #angles = angles - rot
#     # # turn angles into eigenworm coefficient parameters
#     # A = EVecs[:,0:n_coeff]
#     # X = np.linalg.lstsq(A,np.radians(angles),rcond=None)
#     # coeffs = X[0]
#     # params = [x,y,rot,stretch, X[0][0],X[0][1],X[0][2],X[0][3],X[0][4]]

#     # return params

# ##### parameters_to_centeline again
# # copy of parameters_to_centerline

# ### comparison
# cl_orig = copy.copy(centerline)
# cl_orig2 = ps_to_cl_orig(parameters)
# ps2 = cl_to_ps_orig(copy.copy(centerline),EVecs)
# cl_new = ps_to_cl_orig(ps2)
# ps_fixed =  cl_to_ps_fixed(copy.copy(centerline),EVecs)
# cl_fixed = ps_to_cl_orig(ps_fixed)

# #plt.plot(centerline[0],centerline[1],'k.')
# plt.plot(cl_orig[0],cl_orig[1],'b.')
# #plt.plot(cl_orig2[0],cl_orig2[1],'rx')
# plt.plot(cl_new[0],cl_new[1],'r.')
# plt.plot(cl_fixed[0],cl_fixed[1],'gx')








