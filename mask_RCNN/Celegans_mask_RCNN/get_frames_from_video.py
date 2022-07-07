# -*- coding: utf-8 -*-
"""
PDM 19 Jan 2022

This scripts pulls full frames from videos and saves them manual segmentation
 and later training an mRCNN

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# first frame of every video (0-19)
vid_path = r'E:\C. elegans'
vid_list = os.listdir(vid_path)
img_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Celegans\training_set'
if not os.path.exists(img_path+r'\images'):
    os.mkdir(img_path+r'\images')

# i = 0
# for v in range(len(vid_list)):
#     if vid_list[v][-4:] == '.avi':
#         vid = cv2.VideoCapture(vid_path + '\\' + vid_list[v])
#         ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png', img)
#         i += 1

# # frames with nictating animals (0-4)
# vid_file = 'Luca_T2_Rep1_day60002 22-01-18 11-49-24.avi'
# frames = [23,411,1066,1356,1559]
# i = 0
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    

# # frames with nictating animals (5-9)
# vid_file = 'Luca_T2_Rep2_day60004 22-01-18 12-20-36.avi'
# frames = [74,229,427,934,1162]
# i = 5
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    
    
# # frames with nictating animals (10-14)
# vid_file = 'Luca_T2_Rep3_day60001 22-01-18 10-43-45.avi'
# frames = [1471,1765,2308,2610,3070]
# i = 10
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    
    
# # frames with nictating animals (15-19)
# vid_file = 'Luca_T2_Rep4_day60001 22-01-18 11-15-15.avi'
# frames = [141,1096,2330,2668,3719]
# i = 15
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    
    
# # frames with nictating animals (20-24)
# vid_file = 'Luca_T2_Rep3_72h 22-01-15 14-01-07.avi'
# frames = [1,236,435,861,6145]
# i = 20
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    

# # frames with nictating animals (25-29)
# vid_file = 'Luca_T2_Rep2_72h 22-01-15 11-55-21.avi'
# frames = [1,567,1346,1868,3418]
# i = 25
# vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
# for f in frames:
#     vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
#     ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
#     i += 1
    
# added 03-23-2022 to get more challenging examples
# frames with nictating animals and bubbles in arena (30-34)
vid_file = 'Luca_T2_Rep2_day280001 22-02-09 10-04-25.avi'
frames = [1,640,1587,2101,2830]
i = 30
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
    i += 1

# "blown out" looking posts
vid_file = 'Luca_T2_Rep3_60h 22-01-14 22-51-40.avi'
frames = [1,1228,2389,3580,5417]
i = 35
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
    i += 1
    
# lots of missing posts
vid_file = 'Luca_T2_Rep3_day100001 22-01-22 11-01-45.avi'
frames = [1,819,1472,2108,3016]
i = 40
vid = cv2.VideoCapture(vid_path + '\\' + vid_file)
for f in frames:
    vid.set(cv2.CAP_PROP_POS_FRAMES, f-1) # frames are zero indexted in cv2, but not ImageJ
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path+r'\images\Celegans_'+str(i).zfill(5)+'.png',img)
    i += 1
























