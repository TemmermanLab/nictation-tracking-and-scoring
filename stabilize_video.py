# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:35:20 2022

Smooths a video by registering each frame to the previous frame or by
registering each frame to the first frame (register_to_first_frame TRUE). The
latter produces better results in cases tested, but could fail if displacement
is greater than half of the post-to-post distance.

@author: Temmerman Lab
"""

import cv2
import copy
import os
import numpy as np

register_to_first_frame = True
vid_file = r'E:\Celegans_nictation_dataset\Ce_R2_d04.avi'

def stabilize_video(vid_file, out_suffix = '_stabilized_B', 
                    illumination = 'brightfield'):
            
    vid = cv2.VideoCapture(vid_file)

    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, img1 = vid.read(); img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    # registration
    max_corners = 800
    quality = 0.01
    min_dist = 30
    block_sz = 3
    mats = []
    for i in range(n_frames-1):
        print('Registering frame ' + str(i) + ' of ' + str(n_frames) + '.')
        fixed_pts = cv2.goodFeaturesToTrack(img1, max_corners,quality,
                                            min_dist, block_sz)
        
        ret, img2 = vid.read(); img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        moving_pts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2,
                                                           fixed_pts, None)
        idx = np.where(status == 1)[0]
        fixed_pts = fixed_pts[idx]
        moving_pts = moving_pts[idx]
        mat = cv2.estimateAffine2D(moving_pts, fixed_pts)
        mat = mat[0]; mat = np.append(mat,np.array([[0,0,1]]),0)
        mats.append(mat)
        
        if not register_to_first_frame:
            img1 = copy.copy(img2)
        
    
    
    # apply the transformation write stabilized video
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    color = False
    out_file = os.path.splitext(vid_file)[0]+out_suffix+ \
                 os.path.splitext(vid_file)[1]
    v_out = cv2.VideoWriter(out_file, 808466521, vid.get(cv2.CAP_PROP_FPS),
                            (w,h), color)
    
    print('Writing stabilized video...')
    for i in range(n_frames-1):
        ret, img = vid.read(); img = np.squeeze(img[:,:,0])
        if i == 0:
            img_stab = img
        elif i == 1:
            curr_mat = mats[0]
            img_stab = cv2.warpAffine(img, curr_mat[0:2,0:3], (w,h))
        else:
            curr_mat = mats[i-1]
            #curr_mat = np.matmul(curr_mat,mats[i-1])
            img_stab = cv2.warpAffine(img, curr_mat[0:2,0:3], (w,h),
                                      borderValue = 255)
        v_out.write(img_stab)
    
    v_out.release()
    print('Done!')
        


if __name__ == "__main__":
    
    try:
        stabilize_video(vid_file)
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)