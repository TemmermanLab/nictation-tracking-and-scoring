# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:05:32 2023

A GUI for making copies of different frames from different videos and saving
them for manual segmentation and use in training a Mask R-CNN.

@author: PDMcClanahan
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog

import re
non_dec_comma = re.compile(r'[^\d,]+')

import os
import cv2


from pathlib import Path
home = str(Path.home())
sys.path.append(home+r'\Dropbox\Temmerman_Lab\code\tracking-and-scoring-nictation')
import data_management_module as dm



def save_frames(vid, frames, dest):
    # vid = r'E:\Celegans_nictation_dataset\Ce_R1_48h.avi'
    # frames = [1,2,5]
    # dest = r'E:\training_set'
    # halfwidth = 75
        
    v = cv2.VideoCapture(vid)
    
    
    save_path = dest+'\\images'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for f in range(len(frames)):    
        save_name = os.path.split(vid)[1][0:-4]+'_f'+str(frames[f])+'.png'
        v.set(cv2.CAP_PROP_POS_FRAMES, frames[f]-1) # frames are zero indexted in cv2, but not ImageJ
        ret,img = v.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dest+'\\images\\'+save_name,img)
        
    v.release()
    

    
    
    
def make_mask_RCNN_training_set_GUI():
    
    destination = []
    spreadsheet_GUI = tk.Tk()
    
    num_rows = 20
    num_cols = 2
    width_left = 50
    width_right = 50
    vid_box_handles = []
    frame_box_handles = []
    for i in range(num_rows): #Rows
        for j in range(num_cols): #Columns
            if j == 0:    
                vid_box_handles.append(Entry(spreadsheet_GUI, width = width_left,
                                             text=""))
                vid_box_handles[-1].grid(row=i+1, column=j)
            else:
                frame_box_handles.append(Entry(spreadsheet_GUI, width=width_right,
                                               text=""))
                frame_box_handles[-1].grid(row=i+1, column=j)
                
    
    col1_txt = tk.Label(text = 'Video')
    col1_txt.grid(row = 0, column = 0, columnspan = 1, padx = 0, pady = 0)
    
    col2_txt = tk.Label(text = 'Frames')
    col2_txt.grid(row = 0, column = 1, columnspan = 1, padx = 0, pady = 0)
    
    dest_txt = tk.Label(text = 'Destination folder:', anchor = tk.W)
    dest_txt.grid(row = num_rows+1, column = 0, columnspan = 1, padx = 0, pady = 0)
    
        
    def select_video_button():
        a = spreadsheet_GUI.focus_get()
        root = tk.Tk()
        video = tk.filedialog.askopenfilename(initialdir = '/', \
                title = "Select a video from which you want to add frames... \
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
        
        # get video and frame information
        #import pdb; pdb.set_trace()
        videos = []
        frames = []
        for v in range(num_rows):
            if len(vid_box_handles[v].get()) > 0:
                videos.append(vid_box_handles[v].get())
                frame_list = frame_box_handles[v].get()
                frame_list = non_dec_comma.sub('', frame_list)
                frames.append(frame_list.split(','))
                for i in range(len(frames[-1])):
                    frames[-1][i] = int(frames[-1][i])

        # create vignettes in the new training set location
        for v in range(len(videos)):
            save_frames(videos[v], frames[v], destination)
        
        
        
         
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
        make_mask_RCNN_training_set_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
