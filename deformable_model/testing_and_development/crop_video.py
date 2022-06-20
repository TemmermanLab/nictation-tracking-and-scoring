# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:31:34 2021

@author: Temmerman Lab
"""

import cv2
import numpy as np
import os

# debugging
import copy

# vid file is the full path to a video
# frames is a tuple with the first frame, last frame, and (optionally) 
# increment
# frames are considered to be 1 indexed
def crop_video(vid_file, frames, scale = 1.0, save_dir = False):
    
    # import pdb; pdb.set_trace()
    vid = cv2.VideoCapture(vid_file)
    ret,img = vid.read()
    if frames[1] == -1:
        frames[1] = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    # determine if video is really in color (grayscale vids are often read as
    # 3D arrays)
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
        color = True
        R = img[:,:,0]; G = img[:,:,1]; B = img[:,:,2]
        if np.where(R!=G)[0].size == 0 and np.where(R!=B)[0].size == 0 and \
            np.where(B!=G)[0].size == 0:
            color = False
    else:
        color = False
    
    
    # check if the frames requested are present
    num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    if num_f < frames[1]:
        print('WARNING: Video file contains less than '+ str(frames[1])+ \
              ' frames.')
    
            
    # determine the scale of the output
    dim_orig = (img.shape[1],img.shape[0])
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    
    # name the output file
    if scale == 1.0:
        
        if len(frames) >= 3:
            f_to_crop = np.arange(frames[0]-1,frames[1],frames[2])
            out_suffix = '_crop_'+str(frames[0])+'_to_'+str(int(frames[1]))+ \
                '_inc_'+str(int(frames[2]))
        
        else:
            f_to_crop =  np.arange(frames[0]-1,frames[1],1)
            out_suffix = '_crop_'+str(frames[0])+'_to_'+str(int(frames[1]))

    else:
        
        if len(frames) >= 3:
            f_to_crop = np.arange(frames[0]-1,frames[1],frames[2])
            out_suffix = '_crop_'+str(frames[0])+'_to_'+str(int(frames[1]))+ \
                '_inc_'+str(frames[2])+'_scl_'+str(scale)
            
        
        else:
            f_to_crop =  np.arange(frames[0]-1,frames[1],1)
            out_suffix = '_crop_'+str(frames[0])+'_to_'+str(int(frames[1]))+ \
                '_scl_'+str(scale)
            
    if not save_dir:
        out_file = os.path.splitext(video_file)[0]+out_suffix+ \
                 os.path.splitext(video_file)[1]
    else:
        out_file = save_dir + \
            os.path.split(os.path.splitext(video_file)[0])[1] + out_suffix + \
            os.path.splitext(video_file)[1]

    
    # write the cropped video
    #cv2.VideoWriter_fourcc('M','J','P','G') # 1196444237
    #cv2.VideoWriter_fourcc('B','G','R','24') # 844252994
    #vid.get(cv2.CAP_PROP_FOURCC) # 808466521
    v_out = cv2.VideoWriter(out_file, 808466521, \
        vid.get(cv2.CAP_PROP_FPS),(dim[0],dim[1]), color)
    # v_out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M','J','P','G'), \
    #     vid.get(cv2.CAP_PROP_FPS),(dim[0],dim[1]), color)
    print('cropping ',os.path.split(video_file)[1],'...')
    ct = 1
    
    for i in f_to_crop:
        
        #print('writing frame '+str(ct),' of ',str(len(f_to_crop))); ct+=1
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        
        ret,img = vid.read();

        if not ret:
            print('Video not returning frames')
            break
        
        if ~color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if scale != 1.0:
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        v_out.write(img)
    
    v_out.release()

    
    
    

# testing

# video_file = r'E:\C. elegans\Luca_T2_Rep1_day60002 22-01-18 11-49-24.avi'
# frames = [1,300,3]
# crop_video(video_file,frames)

# video_file = r'E:\C. elegans\Luca_T2_Rep3_day100001 22-01-22 11-01-45.avi'
# frames = [1,3,1]
# scale = 1.0
# crop_video(video_file,frames,scale)

# video_file = r'E:\C. elegans\Luca_T2_Rep4_day140001 22-01-26 11-50-52.avi'
# frames = [1,300,3]
# crop_video(video_file,frames)



# video_file = r'C:\Users\Temmerman Lab\Desktop\Bram data\20210923\video_AX7163_B 21-09-23 16-40-40.avi'
# frames = [1,300,3]
# crop_video(video_file,frames)

# making downsampled C. elegans videos for testing
# video_file = r'E:\C. elegans\Luca_T2_Rep1_day60002 22-01-18 11-49-24.avi'
# frames = [1,300,3]
# scale = 0.5
# crop_video(video_file,frames,scale)

# video_file = r'E:\C. elegans\Luca_T2_Rep3_day100001 22-01-22 11-01-45.avi'
# frames = [1,300,3]
# scale = 0.5
# crop_video(video_file,frames,scale)

# video_file = r'E:\C. elegans\Luca_T2_Rep4_day140001 22-01-26 11-50-52.avi'
# frames = [1,300,3]
# scale = 0.5
# crop_video(video_file,frames,scale)


# making downsampled C. elegans videos for tracking
# video_file = r'E:\C. elegans\Luca_T2_Rep1_day60002 22-01-18 11-49-24.avi'
# frames = [1,-1,1]
# scale = 0.5
# crop_video(video_file,frames,scale)

# video_file = r'E:\C. elegans\Luca_T2_Rep3_day100001 22-01-22 11-01-45.avi'
# frames = [1,-1,1]
# scale = 0.5
# crop_video(video_file,frames,scale)

# video_file = r'E:\C. elegans\Luca_T2_Rep4_day140001 22-01-26 11-50-52.avi'
# frames = [1,-1,1]
# scale = 0.5
# crop_video(video_file,frames,scale)

# crop all videos in a directory
try:
    dir_to_crop = 'E:\\20220504_arena_comparison\\'
    save_dir = 'E:\\20220504_arena_comparison_scaled\\'
    #frames = [1,-1,1] # gets modified inside the cropping function(!)
    scale = 0.5
    file_list = os.listdir(dir_to_crop)
    for v in reversed(range(len(file_list))):
        if len(file_list[v]) > 4 and file_list[v][-4:] == '.avi':
            video_file = dir_to_crop + file_list[v]
            frames = [1,-1,1]
            crop_video(video_file,frames,scale,save_dir)
            print(v)
except:
    import pdb
    import sys
    import traceback
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)


