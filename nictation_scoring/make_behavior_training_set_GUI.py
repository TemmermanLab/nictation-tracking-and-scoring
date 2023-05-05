# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:12:41 2023

A GUI for selecting a subset of different tracks from different videos and
then saving the nictation features and vignettes of those tracks in a folder
for manual scoring followed by behavior classifier training.


@author: PDMcClanahan
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog

import re
non_dec_comma = re.compile(r'[^\d,]+')

import os
import cv2
import pandas as pd
import numpy as np

from pathlib import Path
home = str(Path.home())
sys.path.append(home+r'\Dropbox\Temmerman_Lab\code\tracking-and-scoring-nictation')
import data_management_module as dm


def calc_halfwidth(centerlines):
    # calculate the size of window to use based on maximal extent of tracked
    # worms
    extents = np.empty(0)
    for w in range(len(centerlines)):
        for f in range(np.shape(centerlines[w])[0]):
            extent = np.linalg.norm(np.float32(centerlines[w][f][0][0,:])-np.float32(centerlines[w][f][0][-1,:]))
            extents = np.append(extents,extent)
    halfwidth = int(np.max(extents)/1.7)
    #pdb.set_trace()        
    v_out_w = halfwidth*2+1; v_out_h = v_out_w
    
    # set up
    vid = cv2.VideoCapture(vid_path+'\\'+vid_name)
    
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking\\vignettes'
    if not os.path.exists(save_path):
        print('Creating directory for tracking output: '+save_path)
        os.makedirs(save_path)
    is_color = 0
    
    return hw


def make_vignette(vid, track, dest, halfwidth = 100):
    # vid = r'E:\Celegans_nictation_dataset\Ce_R1_48h.avi'
    # track = 3
    # dest = r'E:\training_set'
    # halfwidth = 75
    
    v_out_w = halfwidth*2+1; v_out_h = v_out_w
    
    v_in = cv2.VideoCapture(vid)
    
    
    save_path = dest+'\\vignettes'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    save_name = os.path.split(vid)[1][0:-4]+'_w'+str(track)+'.avi'
    v_out = cv2.VideoWriter(save_path+'\\'+save_name,
        cv2.VideoWriter_fourcc('M','J','P','G'), v_in.get(cv2.CAP_PROP_FPS),
            (v_out_w,v_out_h), False)
    
    centroids_file = vid[0:-4]+"_tracking\\centroids.csv"
    centroids, first_frames = dm.load_centroids_csv(centroids_file)
    
    first = int(first_frames[track])
    last = first+len(centroids[track])
    
    for f in range(last-first):
        msg = 'frame '+str(f+1)+' of '+str(last-first)+', track '+str(track)
        print(msg)
        v_in.set(cv2.CAP_PROP_POS_FRAMES,f+first)
        frame = v_in.read()[1]; frame = frame[:,:,0]
        canvas = np.uint8(np.zeros((np.shape(frame)[0]+halfwidth*2,np.shape(frame)[1]+halfwidth*2)))
        canvas[halfwidth:np.shape(frame)[0]+halfwidth,halfwidth:np.shape(frame)[1]+halfwidth] = frame
        centroid = np.uint16(np.round(centroids[track][f]))
        crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
        v_out.write(crop)
        
    v_out.release()
    

    
    
    
def make_behavior_training_set_GUI():
    
    destination = []
    spreadsheet_GUI = tk.Tk()
    
    num_rows = 20
    num_cols = 2
    width_left = 50
    width_right = 50
    vid_box_handles = []
    track_box_handles = []
    for i in range(num_rows): #Rows
        for j in range(num_cols): #Columns
            if j == 0:    
                vid_box_handles.append(Entry(spreadsheet_GUI, width = width_left,
                                             text=""))
                vid_box_handles[-1].grid(row=i+1, column=j)
            else:
                track_box_handles.append(Entry(spreadsheet_GUI, width=width_right,
                                               text=""))
                track_box_handles[-1].grid(row=i+1, column=j)
                
    
    col1_txt = tk.Label(text = 'Video')
    col1_txt.grid(row = 0, column = 0, columnspan = 1, padx = 0, pady = 0)
    
    col2_txt = tk.Label(text = 'Tracks')
    col2_txt.grid(row = 0, column = 1, columnspan = 1, padx = 0, pady = 0)
    
    dest_txt = tk.Label(text = 'Destination folder:', anchor = tk.W)
    dest_txt.grid(row = num_rows+1, column = 0, columnspan = 1, padx = 0, pady = 0)
    
        
    def select_video_button():
        a = spreadsheet_GUI.focus_get()
        root = tk.Tk()
        video = tk.filedialog.askopenfilename(initialdir = '/', \
                title = "Select a video from which you want to add tracks... \
                ...")
        root.destroy()
        a.delete(0,tk.END)
        a.insert(0,video)
        
    
    def set_destination_button():
        nonlocal destination
        root = tk.Tk()
        destination = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select or create a folder to hold the training set... \
                ...")
        root.destroy()
        dest_txt.config(text = 'Destination folder: ' + destination)
        
    
    def make_training_set_button():
        
        # get video and track information
        #import pdb; pdb.set_trace()
        videos = []
        tracks = []
        for v in range(num_rows):
            if len(vid_box_handles[v].get()) > 0:
                videos.append(vid_box_handles[v].get())
                track_list = track_box_handles[v].get()
                track_list = non_dec_comma.sub('', track_list)
                tracks.append(track_list.split(','))
                for i in range(len(tracks[-1])):
                    tracks[-1][i] = int(tracks[-1][i])
        
        
        # check if the tracking and features are there
        check = True
        for v in videos:
            #pass
            tracking_folder = v[0:-4]+"_tracking/"
            
            # if not os.path.isdir(tracking_folder):
            #     tk.messagebox.showerror(title='Error', \
            #         message='Could not find tracking folder ' + \
            #         os.path.split(v)[1][0:-4] + '_tracking for ' + \
            #         os.path.split(v)[1])
            #     check = False
            
            if not os.path.isfile(tracking_folder+'nictation_features.csv'):
                tk.messagebox.showerror(title='Error', \
                    message='Could not find nictation feature file ' + \
                    os.path.split(v)[1][0:-4] + \
                    '_tracking/nictation_features.csv for ' + \
                    os.path.split(v)[1])
                check = False
            
            if not os.path.isfile(tracking_folder+'centroids.csv'):
                tk.messagebox.showerror(title='Error', \
                    message='Could not find centroids file ' + \
                    os.path.split(v)[1][0:-4] + \
                    '_tracking/centroids.csv for ' + \
                    os.path.split(v)[1])
                check = False
            
            if not os.path.isdir(tracking_folder+'centerlines'):
                tk.messagebox.showerror(title='Error', \
                    message='Could not find centroids file ' + \
                    os.path.split(v)[1][0:-4] + \
                    '_tracking/centroids.csv for ' + \
                    os.path.split(v)[1])
                check = False
        
            
        if check:
            features_list = []
            features_file = destination+'\\nictation_features.csv'
            for v in range(len(videos)):
                tracking_folder = videos[v][0:-4]+"_tracking/"
                features = pd.read_csv(tracking_folder+'nictation_features.csv')
                for t in range(len(tracks[v])):
                    keep = np.zeros(len(features))
                    keep[features['worm']==tracks[v][t]]=1
                    features_keep = features.loc[keep==1]
                    vid_name = os.path.split(videos[v])[1][0:-4]
                    features_keep.insert(0, "vid_name", vid_name, True)
                    features_list.append(features_keep)
            features_training = pd.concat(features_list)
            features_training.to_csv(features_file)
                
            
        
            # create vignettes in the new training set location
            for v in range(len(videos)):
                for t in range(len(tracks[v])):
                    make_vignette(videos[v], tracks[v][t], destination)
        
        
        
         
    def exit_button():
        spreadsheet_GUI.destroy()
        spreadsheet_GUI.quit()
    
    
    
    tk.Button(spreadsheet_GUI,
              text = "SELECT VIDEO",
              command = select_video_button) \
              .grid(row = num_rows+2,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(spreadsheet_GUI,
              text = "SET DESTINATION",
              command = set_destination_button) \
              .grid(row = num_rows+2,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    tk.Button(spreadsheet_GUI,
              text = "MAKE TRAINING SET",
              command = make_training_set_button) \
              .grid(row = num_rows+3,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(spreadsheet_GUI,
              text = "EXIT",
              command = exit_button) \
              .grid(row = num_rows+3,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    curr_row = 0
    
    mainloop()


if __name__ == '__main__':
    
    try:
        make_behavior_training_set_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
