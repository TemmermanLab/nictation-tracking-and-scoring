# -*- coding: utf-8 -*-
"""
PDM 19 Jan 2022

This scripts pulls full frames from videos and saves them manual segmentation
and later training an mRCNN.

"""
# module importation
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# general information
vid_path = r'<your\video\path' # FILL IN with path to your video(s)
vid_list = os.listdir(vid_path)
img_path = r'<your\training\image\path' # FILL IN with path where you want to
# save the video frames
if not os.path.exists(img_path+r'\images'):
    os.mkdir(img_path+r'\images')


   



i = 0 # used to give the frames to segment different names

# description of first video here
vid_file = 'your_video_1.avi' # FILL IN with your video name
frames = [x,y,z,a,b] # FILL IN frames you want to segment for training
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\mRCNN_training_img_'+str(i).zfill(5)+'.png',img)
    i += 1

# description of second video here
vid_file = 'your_video_2.avi' # FILL IN with your video name
frames = [x,y,z,a,b] # FILL IN frames you want to segment for training
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\mRCNN_training_img_'+str(i).zfill(5)+'.png',img)
    i += 1

# description of third video here
vid_file = 'your_video_3.avi' # FILL IN with your video name
frames = [x,y,z,a,b] # FILL IN frames you want to segment for training
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\mRCNN_training_img_'+str(i).zfill(5)+'.png',img)
    i += 1


# copy the blocks of code to add more videos if you wish
