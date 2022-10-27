# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:13:11 2021

This GUI allows click-through execution of the worm tracking code. The
workflow is:
    
    1. Select videos to track (already-selected videos are displayed in the 
                               GUI window)
    2. Set tracking parameters (opens an instance of the parameter GUI)
    3. Run tracking 
    4. Exit

Known issues and improvements:
    -recalculate background when bkgnd_num_frames is changed
    -grab current values when existing from parameter GUI (
        uld be separate
        function for updating params)
    -shrink scale-finding image to fit in the screen
    -automatically set parameters according to scale and brightness
    -load parameters from a previously-tracked video
    -estimate time remaining
    -add scoring
    -option of separate parameters for each video
    -reload parameters previously chosen
    -display tracking
    -improve the speed of the background finding function
    -scale parameter GUI text to make it easy to see at different image
        resolutions

@author: Temmerman Lab
"""


import tkinter as tk
import tkinter.font
from tkinter import ttk
from tkinter import *
import tkinter.filedialog as filedialog # necessary to avoid error
import numpy as np
from PIL import Image, ImageTk
import os
import cv2
import copy
import matplotlib.pyplot as plt
import pandas as pd
import time

import sys
sys.path.append(os.path.split(__file__)[0])

import parameter_GUI
import tracker as tracker
#import data_management_functions as data_f
#import tracking_functions as track_f
#import jan_postprocessing as jan_f
#import cleanup_functions as clean_f

# combining torch with matplotlib or numpy's linalg.lstsq kills the kernel w/o
# this
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# tracking GUI
def tracking_GUI():
    
    # internal functions
    
    # update the information in the main window
    def update_vid_inf(trackers):
        nonlocal vid_inf
        cols = ['name','params?','trcked?','scored?']
        info = np.empty((len(trackers),4),dtype='object')
        for v in range(len(trackers)):
            info[v,0] = trackers[v].vid_name
            if trackers[v].parameters['human_checked']: info[v,1] = 'Yes';
            else: info[v,1] = 'No'
            if hasattr(trackers[v], 'centroids'): info[v,2] = 'Yes';
            else: info[v,2] = 'No'
            if hasattr(trackers[v], 'scores'): info[v,3] = 'Yes';
            else: info[v,3] = 'No'
        
        # clear old text, if any
        vid_inf.delete("1.0","end") 
        
        # print column names
        vid_inf.insert(tk.END, cols[0]+'\t\t\t\t\t\t\t\t\t')
        vid_inf.insert(tk.END, cols[1]+'\t')
        vid_inf.insert(tk.END, cols[2]+'\t')
        vid_inf.insert(tk.END, cols[3]+'\n')
        
        # print video information
        for v in range(len(info)):
            vid_inf.insert(tk.END, info[v,0]+'\t\t\t\t\t\t\t\t\t')
            vid_inf.insert(tk.END, info[v,1]+'\t')
            vid_inf.insert(tk.END, info[v,2]+'\t')
            vid_inf.insert(tk.END, info[v,3]+'\t\n')
        

    
    # load a video or videos to be tracked
    def load_folder():
        nonlocal trackers, data_path
        
        root = tk.Tk()
        data_path = tk.filedialog.askdirectory(initialdir = '/', \
            title = "Select the folder containing videos to be tracked \
            ...")
        root.destroy()
        
        print('Fetching video info '+data_path)
        
        vid_names = os.listdir(data_path)
        for v in reversed(range(len(vid_names))):
            if len(vid_names[v])<4 or vid_names[v][-4:] != '.avi':
                pass
            else:
                trackers.append(tracker.Tracker(data_path+'//'+vid_names[v]))
                
        update_vid_inf(trackers)

    
    # button functions
    
    def load_video_folder_button():
        load_folder()
        
        
    def measure_scale_button():
        nonlocal trackers
        root = tk.Tk()
        scale_file = tk.filedialog.askopenfilename(initialdir = trackers[0].vid_path, \
            title = "Select a video or image to measure the scale \
            ...")
        root.destroy()
        if scale_file[-4:] in ['.avi','.mp4']:
            vid = cv2.VideoCapture(scale_file)
            ret,img = vid.read()
            if len(np.shape(img)) == 3:
                img = np.squeeze(img[:,:,0])
        elif scale_file[-4:] in ['.bmp','.png','.jpg']:
            img = cv2.imread(scale_file,cv2.IMREAD_GRAYSCALE)
        else:
            print('Please choose a supported file (avi, mp4, bmp, png, or jpg)')
            img = []
        
        if len(img) != 0:
            um_per_pix, scale_img = tracker.Tracker.draw_scale(img)
            um_per_pix = round(um_per_pix,3)
            for t in trackers:
                t.parameters['um_per_pix'] = um_per_pix
            


    def set_parameters_button():
        nonlocal trackers
        trackers = parameter_GUI.parameter_GUI(trackers)
        for t in trackers:
            t.save_params()
        update_vid_inf(trackers)
        
        # params['bkgnd_meth'], params['bkgnd_nframes'], params['k_sig'], \
        #     params['k_sz'], params['bw_thr'], params['sz_bnds'], \
        #     params['d_thr'], params['del_sz_thr'], params['um_per_pix'], \
        #     params['min_f'] =  \
        #     parameter_GUI.tracking_param_selector(vid_path +'/'+ vid_names[0])
        
        # vid = data_f.load_video(vid_path +'/'+ vid_names[0])[2]
        # params['fps'] = vid.get(cv2.CAP_PROP_FPS)
        # del vid
        # for v in vid_names:
        #     data_f.save_params_csv(params,vid_path,v)

    
    def track_button():
        for v in range(len(trackers)):
            try:
                trackers[v].track()
            except:
                import pdb
                import sys
                import traceback
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)    
            update_vid_inf(trackers)
        
        print(time.ctime())


    def calculate_features_button():
        for t in trackers:
            t.calculate_features()
    
    
    def score_button():
        for t in trackers:
            t.score_behavior()

         
    def exit_button():
        nonlocal tracking_GUI
        tracking_GUI.destroy()
        tracking_GUI.quit()

    
    # initialize variables
    data_path = []
    trackers = []
    h = 18; w = 100 # in lines and char, based around vid inf window
    
    # set up
    
    # GUI
    tracking_GUI = tk.Tk()
    tracking_GUI.title('Tracking and Scoring GUI')
    tracking_GUI.configure(background = "black")
    # get character size / line spacing in pixels
    chr_h_px = tkinter.font.Font(root = tracking_GUI, font=('Courier',12,NORMAL)).metrics('linespace')
    chr_w_px = tkinter.font.Font(root = tracking_GUI, font=('Courier',12,NORMAL)).measure('m')
    # make the main window as wide and a bit taller than the vid info window
    tracking_GUI.geometry(str(int(w*chr_w_px))+"x"+str(int(chr_h_px*(h+3))))
    
    
    # to do text
    todo_txt = tk.Label(text = 'load a folder containing videos for tracking')
    todo_txt.grid(row = 0, column = 0, columnspan = 6, padx = 0, pady = 0)


    # informational window
    vid_inf = Text(tracking_GUI, height = h, width = w)
    vid_inf.configure(font=("Courier", 12))
    vid_inf.grid(row = 1, column = 0, columnspan = 6, padx = 0, pady = 0)
    
    
    # # buttons
    # tk.Button(tracking_GUI, text = "LOAD VIDEO FOLDER", command = load_video_folder_button, width = 10) .grid(row = 2, column = 0, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    # tk.Button(tracking_GUI, text = "MEASURE SCALE", command = measure_scale_button, width = 10) .grid(row = 2, column = 1, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    # tk.Button(tracking_GUI, text = "SET TRACKING PARAMETERS", command = set_parameters_button, width = 10) .grid(row = 2, column = 2, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    # tk.Button(tracking_GUI, text = "TRACK!", command = track_button,width = 10) .grid(row = 2, column = 3, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    # tk.Button(tracking_GUI, text = "EXIT", command = exit_button,width = 10) .grid(row = 2, column = 4, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')

    # new buttons
    tk.Button(tracking_GUI, text = "LOAD VIDEO FOLDER", command = load_video_folder_button, width = 10) .grid(row = 2, column = 0, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "SET PARAMETERS", command = set_parameters_button, width = 10) .grid(row = 2, column = 1, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "TRACK", command = track_button, width = 10) .grid(row = 2, column = 2, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "CALCULATE FEATURES", command = calculate_features_button,width = 10) .grid(row = 2, column = 3, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "SCORE", command = score_button,width = 10) .grid(row = 2, column = 4, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    tk.Button(tracking_GUI, text = "EXIT", command = exit_button,width = 10) .grid(row = 2, column = 5, padx=0, pady=0, sticky = 'W'+'E'+'N'+'S')
    

    tracking_GUI.mainloop()


if __name__ == '__main__':
    try:
        tracking_GUI()
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    






