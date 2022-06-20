# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:00:40 2022

To do:
    1. Allow user to pick tracked videos
    2. Indicators should be calculated in separate methods
    3. User should be able to save model such that it can be used in tracking
    4. Tracking code should allow the user to pick a different model for head
       tail discrimination.


@author: Temmerman Lab
"""

# Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3 is already
# tracked and manually oriented

# 1. Track Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3.avi
# using current version of the tracker (copied into local folder)
#   i. run tracking_GUI)
#   ii. set parameters equal to those used in Sc_All_smell2...
#   iii. track


# 2. Import modules and define functions needed below

import os
import sys
import csv
import cv2
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dropbox_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab'
#dropbox_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab'


# sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
sys.path.append(dropbox_path + r'\code\nictation')
import tracker_classes as Tracker
import head_tail_scoring_GUI as HT_GUI

sys.path.append(dropbox_path + r'\code\nictation\scripts\20220218_head_tail'+\
                '_orientation\Steinernema_vid_cropped_centerline')
import models_ht as models
    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# methods
def load_centroids(centroids_file):
    xs = []
    ys = []
    first_frames = []
    
    with open(centroids_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                print(f'Column names are {", ".join(row)}')
            elif np.mod(row_count,2)==0:
                # ys.append(np.array(row[1:],dtype='float32'))
                ys.append(np.array(row[1:])[np.where(np.array(row[1:]) != '')[0]].astype('float32'))
            else:
                first_frames.append(int(row.pop(0)))
                # xs.append(np.array(row,dtype='float32'))
                xs.append(np.array(row)[np.where(np.array(row) != '')[0]].astype('float32'))
            row_count += 1
    
    # reshape
    centroids = []
    for t in range(len(xs)):
        centroids.append(np.swapaxes(np.vstack((xs[t],ys[t])),0,1))
    
    return centroids, first_frames


def load_centerlines(centerline_path):
    
    centerlines = []
    centerline_flags = []
    for file in os.listdir(centerline_path):
        if file.endswith(".csv"):
            with open(centerline_path+ "\\" + file) as csv_file:
                
                csv_reader = csv.reader(csv_file, delimiter=',')
                row_count = 0
                coord_list = []
                for row in csv_reader:
                    if row_count == 0:
                        row_count += 1
                    elif row_count == 1:
                        track_flags = np.array(row[1:],dtype=np.int16)
                        row_count +=1
                    else:
                        coord_list.append(np.array(row[1:],dtype='float32'))
                num_pts = int(len(coord_list)/2)
                num_frames = len(coord_list[0])
                
                # shape
                coord_list = np.array(coord_list,dtype = np.float32)
                centerlines_track = np.empty((num_frames,1, num_pts,2),
                                             dtype = np.float32)
                for f in range(num_frames):
                    centerlines_track[f,0,:,0] = coord_list[0:50,f]
                    centerlines_track[f,0,:,1] = coord_list[50:100,f]
                        
            centerlines.append(centerlines_track)
            centerline_flags.append(track_flags)                  
    
    return centerlines, centerline_flags


def load_end_angles(end_angles_file):
    '''Loads the end angles in <end_angles_file>, ignoring the header and worm
    numbers in the first row and first column. Returns a list of numpy arrays
    of the end angles for each frame that worm was tracked.'''
    
    end_angles = []
    with open(end_angles_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count > 0:
                angles = np.array(row[1:], dtype = np.float32)
                end_angles.append(angles)
            row_count += 1

    return end_angles
    

def create_vignettes(vid_name, vid_path, centroids, centerlines, first_frames):
    
    # created cropped videos of each tracked worm, centered around the
    # centroid, thus allowing for rapid loading during manual head / tail and
    # behavioral scoring
    
    # calculate the size of window to use based on maximal extent of tracked
    # worms
    extents = np.empty(0)
    for w in range(len(centerlines)):
        for f in range(np.shape(centerlines[w])[0]):
            extent = np.linalg.norm(np.float32(centerlines[w][f,0,0,:])-np.float32(centerlines[w][f,0,-1,:]))
            extents = np.append(extents,extent)
    halfwidth = int(np.max(extents)/1.7)       
    v_out_w = halfwidth*2+1; v_out_h = v_out_w
    
    # set up
    vid = cv2.VideoCapture(vid_path+'\\'+vid_name)
    
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking\\vignettes'
    if not os.path.exists(save_path):
        print('Creating directory for tracking output: '+save_path)
        os.makedirs(save_path)
    is_color = 0
    
    # create vignettes of each worm
    for w in range(len(centerlines)):
        save_name = 'w'+str(w)+'.avi'
        v_out = cv2.VideoWriter(save_path+ '\\' +save_name, 
                                cv2.VideoWriter_fourcc('M','J','P','G'),
                                vid.get(cv2.CAP_PROP_FPS), (v_out_w,v_out_h),
                                is_color)
        first = first_frames[w]
        last = first_frames[w] + len(centroids[w])
    
        for f in range(first,last):
            msg = 'frame '+str(f-first+1)+' of '+str(last-first)+', track '+str(w)
            print(msg)
            vid.set(cv2.CAP_PROP_POS_FRAMES,f)
            frame = vid.read()[1]; frame = frame[:,:,0]
            canvas = np.uint8(np.zeros((np.shape(frame)[0]+halfwidth*2,np.shape(frame)[1]+halfwidth*2)))
            canvas[halfwidth:np.shape(frame)[0]+halfwidth,halfwidth:np.shape(frame)[1]+halfwidth] = frame
            centroid = np.uint16(np.round(centroids[w][f-first_frames[w]]))
            crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
            v_out.write(crop)
        
        v_out.release()
        
    print('Done!')
    return halfwidth

def save_centroids(centroids, first_frames, save_path, 
                       save_name = 'centroids'):
        
    if not os.path.exists(save_path):
        print('Creating directory for centroids csv and other output: '+save_path)
        os.makedirs(save_path)
    
    save_file_csv = save_path + '\\' + save_name + '.csv'
    
    with open(save_file_csv, mode='w',newline="") as csv_file: 
        
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row = ['First Frame of Track','X and then Y Coordinates on Alternating Rows']
        writer.writerow(row)
        
        for t in range(len(centroids)):
            x_row = [str(first_frames[t])]
            y_row = ['']
            for i in np.arange(0,len(centroids[t])):
                x_row.append(str(round(float(centroids[t][i][0]),1)))
                y_row.append(str(round(float(centroids[t][i][1]),1)))
            writer.writerow(x_row)
            writer.writerow(y_row)
        
    print("Centroids saved as " + save_file_csv )
    
    

def save_centerlines(centerlines, centerline_flags, first_frames, save_path):

    if not os.path.exists(save_path + '\\centerlines'):
        print('Creating directory for centerlines csvs and other output: '+save_path+'\\centerlines')
        os.makedirs(save_path+'\\centerlines')
    
    for w in range(len(centerlines)):
        save_file_csv = save_path + '\\centerlines\\' + 'centerlines_worm_' + "{:06d}".format(w) + '.csv'
        
        with open(save_file_csv, mode='w',newline="") as csv_file: 
            
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # write row 1: frame number
            row = ['frame']
            for f in np.arange(first_frames[w],first_frames[w]+len(centerlines[w])):
                row.append(str(int(f+1)))
            writer.writerow(row)
            
            # write row 2: centerline flag
            row = ['flag']
            for f in np.arange(len(centerline_flags[w])):
                row.append(str(int(centerline_flags[w][f])))
            writer.writerow(row)
            
            # write remaining rows: centerline point coordinates
            for xy in range(2):
                for p in range(np.shape(centerlines[w])[2]):
                    if xy == 0:
                        row = ['x'+str(p)]
                    else:
                        row = ['y'+str(p)]
                    for t in range(len(centerlines[w])):
                        row.append(str(round(float(centerlines[w][t][0,p,xy]),1)))
                    writer.writerow(row)   
            
    print("Centerlines saved in " + save_path + '\\centerlines')


def create_summary_video(vid_file, save_file, centroids, centerlines, first_frames, out_scale = 0.5):
    # setup video
    # out_name = self.save_path + '\\' + os.path.splitext(self.vid_name)[0] \
    #     + '_tracking.avi'
    vid = cv2.VideoCapture(vid_file)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scale)
    out_h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scale)
    v_out = cv2.VideoWriter(save_file,
        cv2.VideoWriter_fourcc('M','J','P','G'),
        vid.get(cv2.CAP_PROP_FPS), (out_w,out_h), 1)
    
    # setup font
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 1.8
    f_thickness = 2
    f_color = (0,0,0)
    
    # loop through frames
    indices = np.linspace(0,num_frames-1,int(num_frames),dtype = 'uint16'); i = 0;
    for i in indices:
        print('Writing frame '+str(int(i+1))+' of '+str(int(num_frames)))
        
        # determine which tracks are present in the frame
        numbers_f = []
        centroids_f = []
        centerlines_f = []
        centerlines_unfixed = []
        centerline_flags = []
        centerline_flags_unfixed = []
        for w in range(len(centroids)):
            if i in np.arange(first_frames[w],first_frames[w]+len(centroids[w])):
                numbers_f.append(w)
                centroids_f.append(centroids[w][i-first_frames[w]])
                centerlines_f.append(centerlines[w][i-first_frames[w]])
                # centerlines_unfixed.append(centerlines_unfixed[w][i-first_frames[w]])
                # centerline_flags.append(centerline_flags[w][i-first_frames[w]])
                # centerline_flags_unfixed.append(centerline_flags_unfixed[w][i-first_frames[w]])
                    
        # load frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_save = np.stack((img,img,img),2)
        
        #import pdb; pdb.set_trace()
        for w in range(len(numbers_f)):
            text = str(numbers_f[w])
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            text_pos = copy.copy(centroids_f[w]) # deepcopy avoids changing objs below
            text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
            text_pos[1] = text_pos[1] + 30
            text_pos = tuple(np.uint16(text_pos))
            img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
            # cline
            
            pts = np.int32(centerlines_f[w][-1])
            pts = pts.reshape((-1,1,2))
            
            img_save = cv2.polylines(img_save, pts, True, (255,0,0), 3)
            img_save = cv2.circle(img_save, pts[0][0], 5, (255,0,0), -1)
        img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
        
        v_out.write(img_save)
    print('DONE')
    v_out.release()





if __name__ == '__main__':
    try:
        make_vignettes = False
        manually_orient = False
        compute_indicators = False
        train_models = True
        calculate_indicators_test = False
        test_model = False
        
        
        # MANUAL SCORING OF H/T ORIENTATION
        
        # centroid_file = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centroids.csv'
        # centerline_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines'
        # vid_file = r'Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi'
        # vid_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline'
        # save_path_centroids = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        # save_path_centerlines = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        
        
        centroid_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centroids.csv'
        centerline_path =dropbox_path +  r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centerlines'
        vid_file = r'Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3.avi'
        vid_path = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline'
        save_path_centroids = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centerlines_manual'
        save_path_centerlines = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centerlines_manual'
        

        if make_vignettes:
            centroids, first_frames = load_centroids(centroid_file)
            centerlines, centerline_flags = load_centerlines(centerline_path)
            halfwidth = create_vignettes(vid_file, vid_path, centroids, centerlines, first_frames)
            print(halfwidth)
        
        if manually_orient:
            
            halfwidth = 188  # Sc_All_smell3_V2
            # halfwidth = 174 # Sc_All_smell2_V2
            
            centroids, first_frames = load_centroids(centroid_file)
            centerlines, centerline_flags = load_centerlines(centerline_path)
            
            centroids_man, centerlines_man, scores = HT_GUI.head_tail_inspector(
                vid_file, vid_path, centroids, centerlines, first_frames,
                halfwidth, scores = -1)
        
            save_centroids(centroids_man, first_frames, save_path_centroids,
                                'centroids')
            save_centerlines(centerlines_man, centerline_flags,
                                  first_frames, save_path_centerlines)
            
            

        
        if compute_indicators:
            # # RELOAD MANUAL CENTERLINES
            # man_centroid_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual\centroids.csv'
            # man_centerline_path = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual\centerlines'
            # vid_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi'
            # profile_save_path = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\troubleshooting'
            # save_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\orientation_indicators.p'
            # end_1_angles_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\end_1_angles.csv' 
            # end_2_angles_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\end_2_angles.csv'
            # scores = np.array([2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3,
            #                         2, 2, 1, 2, 2, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 
            #                         1, 1, 1, 1, 1, 1, 1, 2, 1],dtype = np.uint8)
           
            man_centroid_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centerlines_manual\centroids.csv'
            man_centerline_path = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\centerlines_manual\centerlines'
            vid_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3.avi'
            profile_save_path = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\troubleshooting'
            save_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\orientation_indicators.p'
            end_1_angles_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\end_1_angles.csv' 
            end_2_angles_file = dropbox_path + r'\code\nictation\scripts\20220218_head_tail_orientation\Steinernema_vid_cropped_centerline\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\end_2_angles.csv'
            scores = np.array([2, 1, 1, 3, 1, 2, 2, 3, 2, 1, 1, 1, 1, 2, 3, 3,
                               3, 2, 3, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 3, 3,
                               2, 3, 1, 2, 1, 2, 3, 1, 3, 3, 1, 3, 2, 2, 1, 2,
                               2, 1, 1, 2, 2, 1, 3],dtype = np.uint8)
        
            centroids, first_frames = load_centroids(man_centroid_file)
            centerlines, centerline_flags = load_centerlines(man_centerline_path)
            
            end_1_angles = load_end_angles(end_1_angles_file)
            end_2_angles = load_end_angles(end_2_angles_file)
            
            
            # remove censored centroids and centerlines
            indices = np.where(scores == 3)[0]
            for i in sorted(indices, reverse=True):
                del centerlines[i]
                del centroids[i]
                del first_frames[i]
                scores = np.delete(scores,i)
                del centerline_flags[i]
                del end_1_angles[i]
                del end_2_angles[i]
                
            # save manual scored for later (without censored frames)
            manual_score_file = dropbox_path + '\\code\\nictation\\scripts\\20220218_head_tail_orientation\\Steinernema_vid_cropped_centerline\\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\\manual_orientation_scores.p' 
            pickle.dump(scores, open(manual_score_file, "wb" ) )


            
            # TOTAL MOVEMENT OF EACH END
            def dist(p1,p2):
                d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
                return d
            
            h_t_mov_all = []
            IND_HT_MOV = []
            for clines in centerlines:
                h_t_mov = [0,0]
                for f in range(len(clines)-1):
                    h_t_mov[0] += dist(clines[f][0][0].astype(np.float32),
                                 clines[f+1][0][0].astype(np.float32))
                    h_t_mov[1] += dist(clines[f][0][-1].astype(np.float32),
                                 clines[f+1][0][-1].astype(np.float32))
                h_t_mov_all.append(h_t_mov)
                IND_HT_MOV.append(h_t_mov[0]-h_t_mov[1])
            h_t_mov_all = np.array(h_t_mov_all)
            
            
            fig, axs = plt.subplots(1,1)
            axs.set_aspect('equal','box')
            axs.invert_yaxis()
            axs.plot(centerlines[8][:,0,0,0],centerlines[8][:,0,0,-1],'g-')
            axs.plot(centerlines[8][:,0,-1,0],centerlines[8][:,0,-1,-1],'r-')
            axs.set_title('Worm 8 head / tail path')
            axs.set_xlabel('x')
            axs.set_ylabel('y')
            plt.show()
            
            
            fig, axs = plt.subplots(1,1)
            axs.set_aspect('equal','box')
            errors = 0
            for w in range(len(h_t_mov_all)):
                if h_t_mov_all[w][0] > h_t_mov_all[w][1]:
                    axs.plot(h_t_mov_all[w][0],h_t_mov_all[w][1],'k.',markersize = 0+np.sqrt(len(centroids[w])))
                else:
                    axs.plot(h_t_mov_all[w][0],h_t_mov_all[w][1],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                    errors += 1
            
            axs.text(3000,100,'Error rate: '+str(round(errors / len(h_t_mov_all),2)))
            axs.plot([0,1000],[0,1000],'k:')
            axs.set_title('Total distance')
            axs.set_xlabel('tail')
            axs.set_ylabel('head')
            plt.show()
            
            
            
            # MIDPOINT DIRECTION
            N = np.shape(centerlines[0])[-2]
            fwd_mvnt = []
            bkwd_mvnt = []
            IND_DIR= []
            midpoint = round(N/2)
            for w in range(len(centerlines)):
                clines = centerlines[w]
                fwd_mvnt_w = 0
                bkwd_mvnt_w = 0
                for f in range(len(clines)-1):
                    dir_vect = clines[f,0,midpoint-1]-clines[f,0,midpoint+1]
                    dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                    mov_vect = clines[f+1,0,midpoint]-clines[f,0,midpoint]
                    mvnt_f = np.dot(dir_vect, mov_vect, out=None)
                    if mvnt_f > 0:
                        fwd_mvnt_w += mvnt_f
                    else:
                        bkwd_mvnt_w += abs(mvnt_f)
                fwd_mvnt.append(fwd_mvnt_w)
                bkwd_mvnt.append(bkwd_mvnt_w)
                IND_DIR.append(fwd_mvnt_w-bkwd_mvnt_w)
            
            fig, axs = plt.subplots(1,1)
            axs.set_title('Movement direction')
            axs.set_xlabel('forward movement (pix)')
            axs.set_ylabel('backward movement (pix)')
            axs.set_aspect('equal','box')
            errors = 0
            plt.plot([0,200],[0,200],'k:')
            for w in range(len(centerlines)):
                if fwd_mvnt[w] > bkwd_mvnt[w]:
                    axs.plot(fwd_mvnt[w],bkwd_mvnt[w],'k.',markersize = 0+np.sqrt(len(centroids[w])))
                else:
                    axs.plot(fwd_mvnt[w],bkwd_mvnt[w],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                    errors += 1
            axs.text(-40,200,'Error rate: '+str(round(errors / len(centerlines),2)))
            plt.show()
            
            
            
            # LATERAL MOVEMENT OF THE HEAD AND TAIL
            end_1_lmvnt = []
            end_2_lmvnt = []
            IND_LAT_MOV = []
            for w in range(len(centerlines)):
                clines = centerlines[w]
                end_1_lmvnt_w = 0
                end_2_lmvnt_w = 0
                for f in range(len(clines)-1):
                    # end 1
                    dir_vect = clines[f,0,1]-clines[f,0,0]
                    dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                    mov_vect = clines[f+1,0,0]-clines[f,0,0]
                    end_1_lmvnt_w += abs(np.cross(dir_vect, mov_vect))
                    
                    # end 2
                    dir_vect = clines[f,0,-1]-clines[f,0,-2]
                    dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                    mov_vect = clines[f+1,0,-1]-clines[f,0,-1]
                    end_2_lmvnt_w += abs(np.cross(dir_vect, mov_vect))
                    
                end_1_lmvnt.append(end_1_lmvnt_w)
                end_2_lmvnt.append(end_2_lmvnt_w)
                IND_LAT_MOV.append(end_1_lmvnt_w-end_2_lmvnt_w)
                
            fig, axs = plt.subplots(1,1)
            axs.set_title('Lateral movement')
            axs.set_xlabel('head movement (pix)')
            axs.set_ylabel('tail mov. (pix)')
            axs.set_aspect('equal','box')
            errors = 0
            plt.plot([0,350],[0,350],'k:')
            for w in range(len(centerlines)):
                if end_1_lmvnt[w] > end_2_lmvnt[w]:
                    axs.plot(end_1_lmvnt[w],end_2_lmvnt[w],'k.',markersize = 0+np.sqrt(len(centroids[w])))
                else:
                    axs.plot(end_1_lmvnt[w],end_2_lmvnt[w],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                    errors += 1
            axs.text(2200,1,'Error rate: '+str(round(errors / len(centerlines),2)))
    
            plt.show()
            
            
            # POINTINESS
            fig, axs = plt.subplots(1,1)
            plt.plot([.1,.22],[.1,.22],'k:')
            axs.set_aspect('equal','box')
            axs.set_title('Sharpness of head and tail angles')
            axs.set_xlabel('head angle')
            axs.set_ylabel('tail angle')
            avg_head_angles = []
            avg_tail_angles = []
            IND_POINT = []
            errors = 0
            for w in range(len(end_1_angles)):
                if scores[w] == 1:
                    avg_head_angle = np.mean(end_1_angles[w])
                    avg_tail_angle = np.mean(end_2_angles[w])
                elif scores[w] == 2:
                    avg_tail_angle = np.mean(end_1_angles[w])
                    avg_head_angle = np.mean(end_2_angles[w])
                IND_POINT.append(avg_head_angle- avg_tail_angle)
                if avg_tail_angle < avg_head_angle:
                    axs.plot(-avg_head_angle,-avg_tail_angle,'k.',markersize = 0+np.sqrt(len(centroids[w])))
                else:
                    axs.plot(-avg_head_angle,-avg_tail_angle,'r.',markersize = 0+np.sqrt(len(centroids[w])))
                    errors += 1
                    if avg_head_angle > 0: break
            axs.text(.09,.27,'Error rate: '+str(round(errors / len(centerlines),2)))
            plt.show()
            from scipy import stats
            prob = stats.binom.cdf(errors , len(centerlines), 0.5)
            
            
            # INTENSITY
            med_profiles = []
            vid = cv2.VideoCapture(vid_file)
            for w in range(len(centerlines)):
                shape = np.shape(centerlines[w])
                i_profile = np.empty((shape[2],shape[0]))
                frames = np.linspace(first_frames[w],
                                      first_frames[w]+len(centerlines[w])-1,
                                      len(centerlines[w]))
                for f, cline in enumerate(centerlines[w]):
                    ret = vid.set(cv2.CAP_PROP_POS_FRAMES, frames[f])
                    ret, img = vid.read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    i = cline[0,:,0].astype(np.uint16)
                    j = cline[0,:,1].astype(np.uint16)
                    prof_frame = np.empty(np.shape(cline)[1])
                    for p in range(len(prof_frame)):
                        try:
                            prof_frame[p] = round(np.mean(img[j[p]-1:j[p]+2,i[p] \
                                                             -1:i[p]+2]))
                        except: # in case centerline hits edge of img
                            prof_frame[p] = np.nan

                    # i_profile[:,f] = img[j,i]
                    i_profile[:,f] = prof_frame
                plt.imshow(i_profile)
                plt.show()
                filename = profile_save_path + '//worm_'+str(w)+ \
                           '_intensity.bmp'
                try:
                    ret = cv2.imwrite(filename, i_profile)
                except:
                    pass
                med = np.nanmedian(i_profile,-1)
                med_profiles.append(med)
                
            # overall average profile
            med_med_profile = np.empty((50,len(centroids)))
            for p in range(len(med_profiles)):
                med_med_profile[:,p] = med_profiles[p]
            overall_med = np.median(med_profiles,0)
            
            #np.sqrt(np.sum((avgp - overall_avg)**2)/len(p))
            fig, axs = plt.subplots(1,1)
            axs.plot(med_med_profile,'k-',linewidth = 0.2)
            axs.plot(np.flip(med_med_profile),'r-', linewidth = 0.2)
            axs.plot(overall_med,'k-',linewidth = 1.5)
            axs.plot(np.flip(overall_med),'r-', linewidth = 1.5)
            axs.set_title('Median centerline intensity + reverse')
            axs.set_xlabel('centerline point')
            axs.set_ylabel('intensity')
            #axs.set_aspect('equal','box')
            #errors = 0
    
            
            RMSdiffs = []
            RMSdiffs_rev = []
            IND_INT = []
            for p in range(len(med_profiles)):
                prof = copy.copy(med_profiles[p])
                RMSdiffs.append(np.sqrt(np.sum((prof - overall_med)**2)/
                                        len(prof)))
                prof = np.flip(prof)
                RMSdiffs_rev.append(np.sqrt(np.sum((prof - overall_med)**2)/
                                            len(prof)))
                IND_INT.append(RMSdiffs[-1]-RMSdiffs_rev[-1])
            
            fig, axs = plt.subplots(1,1)
            axs.set_aspect('equal','box')
            errors = 0
            for w in range(len(RMSdiffs)):
                if RMSdiffs[w] < RMSdiffs_rev[w]:
                    axs.plot(RMSdiffs[w],RMSdiffs_rev[w],'k.')
                else:
                    axs.plot(RMSdiffs[w],RMSdiffs_rev[w],'r.')
                    errors += 1
                
            axs.text(20,1,'Error rate: '+str(round(errors / len(med_profiles),
                                                   2)))
            axs.plot([0,40],[0,40],'k:')
            axs.set_title('RMS difference from med. intensity profile')
            axs.set_xlabel('correct')
            axs.set_ylabel('reversed')
            plt.show()
            
            
            # create arrays of the indicators for the forward and backward
            # orientation
            indicator_vals = {
                'direction': IND_DIR,
                'head_tail_movement': IND_HT_MOV,
                'intensity': IND_INT,
                'lateral_movement':IND_LAT_MOV,
                'end_pointiness': IND_POINT
                }

            
            # save indicators
            pickle.dump(indicator_vals, open(save_file, "wb" ) )
        
        if train_models:
            import pdb; pdb.set_trace()
            # load indicators and manual scores
            # indicator_files = [dropbox_path + '\\code\\nictation\\scripts\\20220218_head_tail_orientation\\Steinernema_vid_cropped_centerline\\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\\orientation_indicators.p',
            #                    dropbox_path + '\\code\\nictation\\scripts\\20220218_head_tail_orientation\\Steinernema_vid_cropped_centerline\\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3_tracking\\orientation_indicators.p']
            ind_files = list()
            ind_files.append(dropbox_path + '\\code\\nictation\\scripts\\20'+\
                '220218_head_tail_orientation\\Steinernema_vid_cropped_cent'+\
                'erline\\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_'+\
                'inc_3_tracking\\orientation_indicators.p')
            ind_files.append(dropbox_path + '\\code\\nictation\\scripts\\20'+\
                '220218_head_tail_orientation\\Steinernema_vid_cropped_cent'+\
                'erline\\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_'+\
                'inc_3_tracking\\orientation_indicators.p')
 
            indicator_vals = pickle.load( open( ind_files[0], "rb" ) )
            for f in range(1,len(ind_files)):
                new_vals = pickle.load( open( ind_files[f], "rb" ) )
                for key in indicator_vals.keys():
                    indicator_vals[key] = indicator_vals[key] + new_vals[key]
            
            
            man_sc_fs = list()
            man_sc_fs.append(dropbox_path + '\\code\\nictation\\scripts\\20'+\
                '220218_head_tail_orientation\\Steinernema_vid_cropped_cent'+\
                'erline\\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_'+\
                'inc_3_tracking\\manual_orientation_scores.p')
            man_sc_fs.append(dropbox_path + '\\code\\nictation\\scripts\\20'+\
                '220218_head_tail_orientation\\Steinernema_vid_cropped_cent'+\
                'erline\\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_'+\
                'inc_3_tracking\\manual_orientation_scores.p')
            
            
            manual_scores = pickle.load(open(man_sc_fs[0],"rb" ))
            for f in range(1,len(man_sc_fs)):
                new_scores = pickle.load(open( man_sc_fs[f],"rb" ))
                manual_scores = np.concatenate((manual_scores,new_scores))
                
            # flip scores for centerline that were manually reversed (all
            # indicator values were calculated for manually-oriented
            # centerlines, and all indicators are the difference between the
            # value calculated assuming one orientation and the value
            # calculated assuming the other orientation)
            for key in indicator_vals.keys():
                for i, val in enumerate( indicator_vals[key] ):
                    if manual_scores[i] == 1:
                        pass
                    elif manual_scores[i] == 2:
                        indicator_vals[key][i] = -val
                    elif manual_scores[i] == 3:
                        pass
                    
            
            # put indicator values in a dataframe
            indicator_labels = list(indicator_vals.keys())

            orientation = list(manual_scores)
            for i in range(len(manual_scores)):
                if manual_scores[i] == 1: orientation[i] = 'correct'
                if manual_scores[i] == 2: orientation[i] = 'reversed'
                # if behavior_scores_l[i] == 3: behavior[i] = 'censored'

            data = {'manual_score': manual_scores,'orientation':orientation,
                    indicator_labels[0]:indicator_vals[indicator_labels[0]],
                    indicator_labels[1]:indicator_vals[indicator_labels[1]],
                    indicator_labels[2]:indicator_vals[indicator_labels[2]],
                    indicator_labels[3]:indicator_vals[indicator_labels[3]],
                    indicator_labels[4]:indicator_vals[indicator_labels[4]],
                    }
            
            df = pd.DataFrame(data=data)
            
            # plot normalized indicators by direction
            x_centers = np.linspace(1, 5, 5)
            offset = .25
            half_width = 0.2
            for i, col_name in enumerate(list(df.columns)):
                if i > 1:
                    y_vals = df.loc[df['manual_score']==1,col_name]
                    x_vals = np.linspace(x_centers[i-2]-offset-half_width,x_centers[i-2]-offset+half_width,len(y_vals))
                    plt.plot(x_vals,y_vals,'k.')
                    
                    y_vals = df.loc[df['manual_score']==2,col_name]
                    x_vals = np.linspace(x_centers[i-2]-offset-half_width,x_centers[i-2]-offset+half_width,len(y_vals))
                    plt.plot(x_vals,y_vals,'r.')
            plt.show()
                
                print(i)
                print(col_name)
            
            
            # shuffle data into training and test groups
            X_train, X_test, y_train, y_test = models.shuffle(df)
            
            # train a series of models on shuffled data
            model_types = ['logistic regression','decision tree',
                'k nearest neighbors','linear discriminant analysis',
                'Gaussian naive Bayes','support vector machine',
                'random forest']

            heatmap = np.empty((len(model_types),2))
            
            for mtype in range(len(model_types)):
                model, train_acc, test_acc, probabilities, predictions = \
                    models.learn_and_predict(X_train, X_test, y_train, y_test,
                                             model_types[mtype])
                heatmap[mtype,0:2] = [train_acc,test_acc]
                

            # show a heatmap of model performance
            fig, axes = plt.subplots()
            im = axes.imshow(heatmap,cmap='viridis')
            plt.title('Model Performance')
            axes.xaxis.set_label_position('top')
            axes.xaxis.tick_top() 
            axes.set_xticks([0,1])
            axes.set_xticklabels(['train','test'])
            axes.set_yticks(np.arange(len(model_types)))
            axes.set_yticklabels(model_types)
            #axes.set_xlabel('shuffled         split')
            plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center',
                     rotation_mode = 'anchor')
            
            for i in range(2):
                for j in range(len(model_types)):
                    text = axes.text(i,j,"%0.2f" % heatmap[j,i],ha='center',
                                     va='center',fontweight = 'bold')
            plt.show()
            
            sdfssd
            
            # show a heatmap of metric performance with the random forest
            model, train_acc, test_acc, probabilities, predictions = \
            models.learn_and_predict(X_train, X_test, y_train, 
            y_test,'random forest')


            # performance on test set
            test_perf = model.score(X_test,y_test) # same as test_acc above
            
            # performance on test set with one scrambled
            perfs_removed = []
            for col in X_test.columns:
                X_test_scram = copy.deepcopy(X_test)
                scram_col = np.array(X_test_scram[col])
                np.random.shuffle(scram_col)
                X_test_scram[col] = scram_col
                perfs_removed.append(model.score(X_test_scram,y_test))
            decrements = perfs_removed-test_perf
            
            # performance on a test set with total and lateral head movement
            # scrambled
            X_test_scram = copy.deepcopy(X_test)
            for col in X_test.columns[[1,3]]:
                scram_col = np.array(X_test_scram[col])
                np.random.shuffle(scram_col)
                X_test_scram[col] = scram_col
            perf_all_head_mov_scram = model.score(X_test_scram,y_test)
            
            # performance on totally-scrambled test set
            X_test_scram = copy.deepcopy(X_test)
            for col in X_test.columns:
                scram_col = np.array(X_test_scram[col])
                np.random.shuffle(scram_col)
                X_test_scram[col] = scram_col
            perf_all_scram = model.score(X_test_scram,y_test)
            
            # performance on test set with one metric unscrambled
            perfs_added = []
            for col in X_test.columns:
                X_test_unscram = copy.deepcopy(X_test_scram)
                X_test_unscram[col] = X_test[col]
                perfs_added.append(model.score(X_test_unscram,y_test))
            
            increments = perfs_added-perf_all_scram
            
            # plot indicator performance heatmap
            heatmap2 = np.vstack((decrements, increments))
            fig, axes = plt.subplots()
            im = axes.imshow(heatmap2,cmap='RdBu')
            plt.title('Contribution of Metrics to Random Forest Performance')
            axes.xaxis.set_label_position('bottom')
            axes.set_xticks(np.arange(len(X_test.columns)))
            axes.set_xticklabels(X_test.columns,rotation = 45, ha="right")
            axes.set_yticks([0,1])
            axes.set_yticklabels(['decrement','increment'])
            fw = ['normal','normal']
            for i in range(2):
                for j in range(len(X_test.columns)):
                    text = axes.text(j,i,"%0.2f" % heatmap2[i,j],ha='center',va='center',fontweight = fw[i],fontsize=5)
            plt.show()
            
            heatmap3 = np.vstack((decrements[0:13], increments[0:13]))
            fig, axes = plt.subplots()
            im = axes.imshow(heatmap3,cmap='RdBu')
            plt.title('Contribution of Metrics to Random Forest Performance')
            axes.xaxis.set_label_position('bottom')
            axes.set_xticks(np.arange(len(X_test.columns[0:13])))
            axes.set_xticklabels(X_test.columns[0:13],rotation = 45, ha="right")
            axes.set_yticks([0,1])
            axes.set_yticklabels(['decrement','increment'])
            fw = ['bold','bold']
            for i in range(2):
                for j in range(len(X_test.columns[0:13])):
                    if i == 0:
                        text = axes.text(j,i,"%0.2f" % abs(heatmap3[i,j]),ha='center',va='center',fontweight = fw[i],fontsize=10,color = [0,0,0],rotation = 0)
                    else:
                        text = axes.text(j,i,"%0.2f" % abs(heatmap3[i,j]),ha='center',va='center',fontweight = fw[i],fontsize=10,color = [0,0,0],rotation = 0)
            plt.show()
            
            
            plt.scatter(increments,abs(decrements),color = 'k')
            plt.title('Increment vs. Decrement by Indicator')
            plt.xlabel('increment')
            plt.ylabel('decrement')
            plt.show
                        
        
        if calculate_indicators_test:
            pass
            # calculate the same indicators for a different video
            
        if test_model:
            pass
            # test the performance of the combination of indicators on the test
            # video
        
        
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
# 3. Manually orient the centerlines
