# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:09:24 2021

This script scales down the training images and masks while preserving the
mask pixel values (naive rescaling would give them blurry edges with incorrect
grayscale values)

@author: Temmerman Lab
"""

import cv2
import os
import matplotlib.pyplot as plt

# user inputs
original_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Celegans\training_set'
rescaled_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Celegans\training_set_scaled'
scale_prop = .5 # percent of original size


img_list = os.listdir(original_dir +'\images')
mask_list = os.listdir(original_dir + '\masks')

for i in range(len(img_list)):
    img = cv2.imread(original_dir+'\\images\\'+img_list[i],cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(original_dir+'\\masks\\'+mask_list[i],cv2.IMREAD_GRAYSCALE)
    
    width = int(img.shape[1] * scale_prop)
    height = int(img.shape[0] * scale_prop)
    dim = (width, height)
    
    img_scaled = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    mask_scaled = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
    
    cv2.imwrite(rescaled_dir+'\\images\\'+img_list[i],img_scaled)
    cv2.imwrite(rescaled_dir+'\\masks\\'+mask_list[i],mask_scaled)





