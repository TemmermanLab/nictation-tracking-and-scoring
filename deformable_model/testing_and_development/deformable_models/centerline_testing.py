# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:17:19 2021

@author: Temmerman Lab
"""

import cv2
import sys
import os
import numpy as np
sys.path.append(os.path.split(__file__)[0])

import tracker_classes as tracker
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import matplotlib.pyplot as plt

bw_files = [
    # r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw1.png',
    # r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw2.png',
    # r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw3.png',
    # r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw4.png',
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw5.png',
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw6.png',
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw7.png',
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw8.png',
    ]


for i in bw_files:

    bw = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
    
    for i in range(4):
        bw = np.rot90(bw)
        if __name__ == '__main__':
            try:
                centerline = tracker.Tracker.find_centerline(bw)
            except:
                import pdb
                import sys
                import traceback
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
                
        
        plt.figure()
        plt.imshow(bw,cmap = 'gray')
        plt.plot(centerline[:,0],centerline[:,1],'r.')
        plt.show()