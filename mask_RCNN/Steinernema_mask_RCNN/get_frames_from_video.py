# -*- coding: utf-8 -*-
"""
PDM 28 Nov 2021

This scripts pulls full frames from videos and saves them manual segmentation and later training an mRCNN



"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# first frame of every video (0-19)
vid_path = r'D:\Steinernema'
vid_list = os.listdir(vid_path)
img_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20211128_full_frame_Steinernema_segmentation\dataset'
if not os.path.exists(img_path+r'\images'):
    os.mkdir(img_path+r'\images')

i = 0
for v in range(len(vid_list)):
    if vid_list[v][-4:] == '.avi':
        vid = cv2.VideoCapture(vid_path + '\\' + vid_list[v])
        ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_path+r'\images\fftsSc_'+str(i).zfill(5)+'.png', img)
        i += 1

# frames with nictating animals (20-24)
vid_path = r'D:\Steinernema'
vid_file = 'Sc_All_smell1_V2_ 21-09-16 18-34-20.avi'
frames = [287,480,1487,1628,1746]
i = 20
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\fftsSc_'+str(i).zfill(5)+'.png',img)
    i += 1
    

# frames with nictating animals (25-29)
vid_path = r'D:\Steinernema'
vid_file = 'Sc_All_smell2_V2_ 21-09-17 14-51-41.avi'
frames = [193,493,598,1392,1724]
i = 25
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\fftsSc_'+str(i).zfill(5)+'.png',img)
    i += 1
    
    
    # frames with nictating animals (30-34)
vid_path = r'D:\Steinernema'
vid_file = 'Sc_All_smell3_V2_ 21-09-17 15-26-15.avi'
frames = [221,501,1441,2003,2234]
i = 30
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\fftsSc_'+str(i).zfill(5)+'.png',img)
    i += 1

























