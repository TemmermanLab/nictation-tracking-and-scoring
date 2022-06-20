# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:25:20 2021

@author: Temmerman Lab
# """

# otsu method on mask RCNN output... tends to choose a way-too-high threshold

import cv2
img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame_mRCNN_grayscale_mask.png'
gray = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
gray = cv2.GaussianBlur(gray,(11,11),2,cv2.BORDER_REPLICATE)

from skimage import filters
thr = filters.threshold_otsu(img)
bw = copy.copy(gray); bw[bw>=thr] = 255; bw[bw<thr] = 0;

import matplotlib.pyplot as plt
plt.imshow(gray,cmap = 'gray'); plt.axis('off'); plt.show()
plt.imshow(bw,cmap = 'gray'); plt.axis('off'); plt.show()
cv2.imwrite(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame_mRCNN_otsu_mask_smooth.png',bw)


# using an threshold that seems right

import cv2
img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame_mRCNN_grayscale_mask.png'
gray = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
gray = cv2.GaussianBlur(gray,(11,11),1.5,cv2.BORDER_REPLICATE)

from skimage import filters
thr = 50
bw = copy.copy(gray); bw[bw>=thr] = 255; bw[bw<thr] = 0;

import matplotlib.pyplot as plt
plt.imshow(gray,cmap = 'gray'); plt.axis('off'); plt.show()
plt.imshow(bw,cmap = 'gray'); plt.axis('off'); plt.show()
cv2.imwrite(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame_mRCNN_arb_mask_smooth.png',bw)


# otsu method on regular 



