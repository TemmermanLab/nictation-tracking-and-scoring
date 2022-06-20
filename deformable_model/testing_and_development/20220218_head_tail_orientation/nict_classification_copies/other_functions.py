# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 10:13:19 2021

@author: PDMcClanahan
"""

import cv2
import numpy as np


def get_background(vid, bkgnd_nframes = 10, method = 'max_merge'):
    supported_meths = ['max_merge','min_merge']
    #pdb.set_trace()
    if method not in supported_meths:
        raise Exception('Background method not recognized, method must be one of ' + str(supported_meths))
    else:
        print('Calculating background image...')
        num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        inds = np.round(np.linspace(0,num_f-1,bkgnd_nframes)).astype(int)
        for ind in inds:
            vid.set(cv2.CAP_PROP_POS_FRAMES, ind)
            success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ind == inds[0]:
                img = np.reshape(img,(img.shape[0],img.shape[1],1)) 
                stack = img
            else:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
                stack = np.concatenate((stack, img), axis=2)
        if method == 'max_merge':
            bkgnd = np.amax(stack,2)
        elif method == 'min_merge':
            bkgnd = np.amin(stack,2)
        print('Background image calculated')
    return bkgnd 
    
    