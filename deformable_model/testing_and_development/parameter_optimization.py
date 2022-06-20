# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:39:45 2021

This code takes a background-subtracted or Mask R-CNN generated grayscale
image and applies smoothing and thresholding to it to binarize any objects it
contains. The result is compared to a manually-segmented image using IoU. This
is done with a range of smoothing sigmas and thresholds in order to find the
best parameters.

@author: P. D. McClanahan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def optimize_binarization(img, mask, show):
    
    # set ranges of sigmas and thresholds to check
    sigmas = np.arange(0,5,.05)
    threshes = np.arange(1,255,1)
    IoU_matrix = np.empty((len(sigmas),len(threshes)))
    
    # calculate IoU for each parameter pair
    for s in range(len(sigmas)):
        #print(s)
        for t in range(len(threshes)):
            sigma = sigmas[s]
            thresh = threshes[t]
            k_size = (round(sigma*3)*2+1,round(sigma*3)*2+1)
            if k_size[0]<5:
                k_size = (5,5)
            
            smooth = cv2.GaussianBlur(img,k_size,sigma,cv2.BORDER_REPLICATE)
            trash,bw = cv2.threshold(smooth,thresh,255,cv2.THRESH_BINARY)
            
            bw = bw/255
            
            I = len(np.where(mask+bw == 2)[0])
            U = len(np.where(mask+bw > 0)[0])
            IoU_matrix[s,t] = I/U

    #import pdb; pdb.set_trace()
    # plot IoU heatmap with optimal parameters
    if show:
        plt.figure(figsize=(5,5))
        plt.imshow(IoU_matrix)
        x_inds = np.arange(4,254,25)
        x_labels = threshes[x_inds]
        plt.xticks(x_inds,x_labels)
        y_inds = np.arange(0,len(sigmas),round(len(sigmas)/10.0))
        y_labels = sigmas[y_inds]
        plt.yticks(y_inds,y_labels)
        plt.xlabel('bw threshold')
        plt.ylabel('smoothing sigma')
        plt.title('IoU')
        s,t = np.where(IoU_matrix == np.max(IoU_matrix))
        if len(s) > 1:
            print('Warning: tie for optimal parameters')
        s = s[0]; t = t[0] # would be better to take the middle value in the case of a tie
        plt.plot(t,s,'rx')
        plt.show()
    
    sigma = sigmas[s]
    k_size = (round(sigma*3)*2+1,round(sigma*3)*2+1)
    thresh = int(threshes[t])
    smooth = cv2.GaussianBlur(img,k_size,sigma,cv2.BORDER_REPLICATE)
    trash,bw = cv2.threshold(smooth,thresh,255,cv2.THRESH_BINARY)
    best_iou = np.max(IoU_matrix)

    return sigma, thresh, best_iou, bw


import traceback, sys, code
import pdb

if __name__ == '__main__':
    try:    
        worm_img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\img.png'
        mask_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\img_mask.png'
        rcnn_output_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\mRCNN_output.png'
        bkgnd_sub_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\img_bkgnd.png'
        
        img = cv2.imread(worm_img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        rcnn_output = cv2.imread(rcnn_output_file, cv2.IMREAD_GRAYSCALE)
        bkgnd_sub = cv2.imread(bkgnd_sub_file, cv2.IMREAD_GRAYSCALE)
        
        mask[np.where(mask != 0)] = 1
        
        sigma_rcnn, thresh_rcnn, iou_rcnn, bw_rcnn = optimize_binarization(rcnn_output,mask,True)
        sigma_inten, thresh_inten, iou_inten, bw_inten = optimize_binarization(bkgnd_sub,mask,True)
    
        plt.imshow(mask,cmap = 'gray')
        plt.title('manual segmentation')
        plt.axis('off')
        plt.show()
        
        plt.imshow(bw_rcnn,cmap = 'gray')
        plt.title('mask R-CNN segmentation, IoU = '+str(round(iou_rcnn,3)))
        plt.axis('off')
        plt.show()
        
        plt.imshow(bw_inten,cmap = 'gray')
        plt.title('intensity-based segmentation, IoU = '+str(round(iou_inten,3)))
        plt.axis('off')
        plt.show()
    except:
        
        # normal version
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
        # # code.interact
        # type, value, tb = sys.exc_info()
        # traceback.print_exc()
        # last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        # frame = last_frame().tb_frame
        # ns = dict(frame.f_globals)
        # ns.update(frame.f_locals)
        # code.interact(local=ns)







