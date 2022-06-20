# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:52:38 2022

@author: Temmerman Lab
"""

import os
import sys
import copy
import numpy as np
#sys.path.append(os.path.split(__file__)[0])'
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\nictation')
import tracker_classes as tracker
import deformable_worm_module as defmod
import matplotlib.pyplot as plt
from scipy import interpolate

os.environ['KMP_DUPLICATE_LIB_OK']='True'
t = tracker.Tracker(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_cropped\super_crop.avi')

try:
    t.track(True)
except:
    import pdb
    import sys
    import traceback
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)       
    
# # problematic
# centerline = np.empty((2,50))
# centerline[0]= np.array([225, 221, 216, 211, 206, 201, 195, 190, 185, 179, 175, 172, 169,
#    166, 163, 163, 162, 162, 163, 169, 174, 179, 184, 189, 195, 200,
#    205, 210, 216, 221, 226, 231, 235, 239, 243, 247, 251, 254, 257,
#    261, 265, 270, 276, 281, 286, 291, 297, 300, 301, 303],
#   np.uint16)
# centerline[1] = np.array([240, 241, 243, 244, 246, 247, 247, 247, 247, 246, 244, 240, 235,
#    231, 226, 221, 216, 211, 207, 206, 205, 204, 203, 202, 201, 200,
#    199, 200, 200, 200, 199, 197, 194, 191, 187, 184, 179, 175, 171,
#    167, 165, 166, 167, 168, 169, 170, 171, 171, 166, 161],
#   np.uint16)

# # wormm 1 frame 1 from super crop
# centerline = np.empty((2,50))
# centerline[0] = [2129,2135,2141,2147,2153,2159,2165,2171,2177,2182,2185,2189,
# 2191,2193,2194,2192,2191,2188,2185,2181,2177,2172,2168,2163,2159,2156,2153,
# 2150,2148,2147,2146,2146,2146,2147,2147,2147,2147,2148,2148,2150,2152,2155,
# 2159,2163,2169,2174,2178,2182,2187,2192]
# centerline[0] = centerline[0] - np.mean(centerline[0]) + 200
# centerline[1] = [107,106,104,103,100,100,101,101,103,107,112,117,123,129,135,
#                  142,148,153,159,163,168,173,176,181,186,191,196,202,208,214,
#                  221,227,233,239,246,252,258,265,271,277,283,289,294,298,300,
#                  304,309,314,317,321]
# centerline[0] = centerline[0] - np.mean(centerline[0]) + 200



# # change the centerline from 50 points to 100
# tck, u = interpolate.splprep([centerline[0],centerline[1]], s=0)
# unew = np.linspace(0,1,100)
# centerline_100 = np.empty((2,100),np.float32)
# centerline_100[0],centerline_100[1] = np.array(interpolate.splev(unew,tck))
# plt.plot(centerline[0],centerline[1],'r.')
# plt.plot(centerline_100[0],centerline_100[1],'g.')
# plt.show()


# # set up model
# mod = defmod.Eigenworm_model()
# mod.set_centerline(centerline_100)
# mod.centerline_to_parameters()
# mod.curb_parameters()
# # self.update_bw_image()


# # show bw image


# # # recalculate the centerline and bw image to reflect the fitted
# # # parameters
# self.parameters_to_centerline()
# self.update_bw_image()




# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# mod = defmod.Eigenworm_model()
# mod.set_centerline(centerline)

# plt.imshow(mod.bw_image,cmap= 'gray')
# plt.plot(centerline[0],centerline[1],'r:')
# plt.show()

# params = copy.copy(mod.parameters)
# mod.parameters_to_centerline()
# mod.update_bw_image()



# # try:
    
    
    
    
# # except:
# #     import pdb
# #     import sys
# #     import traceback
# #     extype, value, tb = sys.exc_info()
# #     traceback.print_exc()
# #     pdb.post_mortem(tb)
    
    
    
    
    