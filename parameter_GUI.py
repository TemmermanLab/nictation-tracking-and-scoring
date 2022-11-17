# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:55:41 2021

Takes a list of tracker objects and allows the user to set parameters for
intensity based segmentation. Those parameters are changed in the tracker
objects and saved in .csv format.

Issues / improvements:
    -scale calculating function
    -pre-calculate parameters based on scale and stage being imaged
    -would be nice to cycle back and forth
    -steps are not labeled
    -no way to zoom image
    -possibly helpful info: 
    -no way to change frame being inspected
    -cartoon showing worm size and distance threshold, and size threshold
     would help
    -ensure a the new mask RCNN is used when the model file is changed
    

@author: PDMcClanahan
"""

import cv2
import numpy as np
import os
import pdb, traceback, sys
from PIL import Image as Im, ImageTk
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import *

import sys
sys.path.append(os.path.split(__file__)[0])
import tracker as tracker

# find directories for mRCNN files and behavior model files
mrcnn_path = os.path.split(__file__)[0] + '\\mask_RCNN'
behavior_model_path = os.path.split(__file__)[0] + \
    '\\nictation_scoring\models'


def parameter_GUI(trackers):
    
    # vars for display
    bkgnd_meths = ('max_merge','min_merge')
    img_type_inds = [0,1,2,3,4,5]
    img_type_ind = 5
    f = 0
    v = 0

    # init outcome values
    if trackers[v].segmentation_method == 'intensity':
        bkgnd = trackers[v].background
    f_width = 600
    f_height = int(np.round(600*(
        trackers[v].dimensions[0]/trackers[v].dimensions[1])))
    img, diff, smooth, bw, bw_sz, final = trackers[v].show_segmentation(v)
    
    
    def choose_mrcnn_file_button():
        nonlocal trackers
            
        root = tk.Tk()
        mrcnn_file = tk.filedialog.askopenfilename(initialdir = mrcnn_path, \
            title = "Select a mask R-CNN file (.pt) for tracking\
            ...")
        root.destroy()
        
        for t in trackers:
            t.parameters['mask_RCNN_file'] = mrcnn_file
        
        enter_mrcnn_file.delete(0, 'end')
        enter_mrcnn_file.insert(0,trackers[v].parameters['mask_RCNN_file'])
    
    
    def choose_behavior_model_button():
        nonlocal trackers
            
        root = tk.Tk()
        behavior_model_file = tk.filedialog.askopenfilename(initialdir = \
            behavior_model_path, title = \
            "Select a model (.pkl) for scoring behavior...")
        root.destroy()
        
        for t in trackers:
            t.parameters['behavior_model_file'] = behavior_model_file
        
        enter_behavior_model_file.delete(0, 'end')
        enter_behavior_model_file.insert(
            0,trackers[v].parameters['behavior_model_file'])
            
            
    def update_images_button():
        print('Updating images...')
        nonlocal trackers, v, f
        nonlocal img, diff, smooth, bw, bw_sz, final
        
        try:
            v = int(enter_video_num.get())-1
            f = int(enter_frame_num.get())-1
            trackers[v].parameters['human_checked'] = True
            
            if enter_bkgnd_meth.get() != trackers[v].parameters['bkgnd_meth']:
                trackers[v].parameters['bkgnd_meth'] = enter_bkgnd_meth.get()
                trackers[v].get_background()
            trackers[v].parameters['bkgnd_nframes'] = \
                int(enter_bkgnd_nframes.get())
            trackers[v].parameters['k_sig'] = float(enter_k_sig.get())
            trackers[v].parameters['min_f'] = int(enter_min_f.get())
            trackers[v].parameters['bw_thr'] = int(enter_bw_thr.get())
            trackers[v].parameters['d_thr'] = int(enter_d_thr.get())
            trackers[v].parameters['area_bnds'] = \
                (int(enter_min_sz.get()),int(enter_max_sz.get()))
            trackers[v].parameters['mask_RCNN_file'] = str(
                enter_mrcnn_file.get())
            trackers[v].parameters['behavior_model_file'] = str(
                enter_behavior_model_file.get())
            
        except:
            print('Enter numbers in all the fields')

        if enter_um_per_pix.get() == '' or enter_um_per_pix.get() == 'None':
            trackers[v].parameters['um_per_pix'] = 'None'
        else:
            trackers[v].parameters['um_per_pix'] = \
                float(enter_um_per_pix.get())
        
        if v < len(trackers) and f < trackers[v].num_frames:
            img, diff, smooth, bw, bw_sz, final = \
                trackers[v].show_segmentation(f)
            update_win(img_type_ind)

        else:
            print('Enter in range video and frame numbers')
      
            
    def cycle_image_button():
        nonlocal img_type_ind
        img_type_ind = img_type_ind+1
        if img_type_ind >= len(img_type_inds):
            img_type_ind = 0
        print(img_type_ind)
        update_win(img_type_ind)
      
        
    def find_scale_button():
        nonlocal trackers
        nonlocal enter_um_per_pix
        root = tk.Tk()
        scale_file = tk.filedialog.askopenfilename(
            initialdir = trackers[0].vid_path, 
            title = "Select a video or image to measure the scale...")
        root.destroy()
        if scale_file[-4:] in ['.avi','.mp4']:
            vid = cv2.VideoCapture(scale_file)
            ret,img = vid.read()
            if len(np.shape(img)) == 3:
                img = np.squeeze(img[:,:,0])
        elif scale_file[-4:] in ['.bmp','.png','.jpg']:
            img = cv2.imread(scale_file,cv2.IMREAD_GRAYSCALE)
        else:
            print(
                'Please choose a supported file (avi, mp4, bmp, png, or jpg)')
            img = []
        
        if len(img) != 0:
            um_per_pix, scale_img = tracker.Tracker.draw_scale(img)
            um_per_pix = round(um_per_pix,3)
            for t in trackers:
                t.parameters['um_per_pix'] = um_per_pix
            enter_um_per_pix.delete(0, 'end')
            enter_um_per_pix.insert(0,trackers[v].parameters['um_per_pix'])
       
        
    def save_exit_button():
        
        ### move this under an "apply all" button in the future
        nonlocal trackers
        for t in trackers:
            try:
                if enter_bkgnd_meth.get() != t.parameters['bkgnd_meth']:
                    t.parameters['bkgnd_meth'] = enter_bkgnd_meth.get()
                    t.get_background()
                t.parameters['bkgnd_nframes'] = int(enter_bkgnd_nframes.get())
                t.parameters['k_sig'] = float(enter_k_sig.get())
                t.parameters['min_f'] = int(enter_min_f.get())
                t.parameters['bw_thr'] = int(enter_bw_thr.get())
                t.parameters['d_thr'] = int(enter_d_thr.get())
                t.parameters['area_bnds'] = (int(enter_min_sz.get()),
                                             int(enter_max_sz.get()))
            except:
                print('Enter numbers in all the fields')
            
            if enter_um_per_pix.get() == '':
                t.parameters['um_per_pix'] = 'None'
            else:
                try:
                    t.parameters['um_per_pix'] = float(enter_um_per_pix.get())
                except:
                    t.parameters['um_per_pix'] = enter_um_per_pix.get()
        
        param_insp.destroy()
        param_insp.quit()

    
    def update_win(img_type_ind):
        nonlocal frame
        
        print(img_type_ind)
        if  img_type_ind == 0:
            frame = img
        elif img_type_ind == 1:
            frame = diff
        elif img_type_ind == 2:
            frame = smooth
        elif img_type_ind == 3:
            frame = bw
        elif img_type_ind == 4:
            frame = bw_sz
        elif img_type_ind == 5:
            frame = final
        
        frame = Im.fromarray(frame)
        frame = frame.resize((f_width,f_height),Im.NEAREST)
        frame = ImageTk.PhotoImage(frame)
        img_win.configure(image = frame)
        img_win.update()
    
        
    # set up GUI
    param_insp = tk.Toplevel()
    param_insp.title('Tracking Parameter Inspection GUI')
    param_insp.configure(background = "black")
    
    # set up video window
    img, diff, smooth, bw, bw_ws, final = trackers[v].show_segmentation(f)
    frame = Im.fromarray(final)
    frame = frame.resize((f_width,f_height),Im.NEAREST)
    frame = ImageTk.PhotoImage(frame)
    img_win = Label(param_insp,image = frame, bg = "black")
    img_win.grid(row = 0, column = 0, columnspan = 4, padx = 0, pady = 0)
    
    # set up text and input windows
    Label (param_insp,text="Video:", bg = "gray", fg = "black") \
        .grid(row = 1, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_video_num = Entry(param_insp, bg = "white")
    enter_video_num.grid(row = 1, column = 1,padx=1, pady=1, sticky = W+E)
    enter_video_num.insert(0,str(v+1))
    
    
    Label (param_insp,text="Frame:", bg = "gray", fg = "black") \
        .grid(row = 1, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_frame_num = Entry(param_insp, bg = "white")
    enter_frame_num.grid(row = 1, column = 3,padx=1, pady=1, sticky = W+E)
    enter_frame_num.insert(0,str(f+1))
    
    
    Label (param_insp,text="Background method (max_merge or min_merge):", 
           bg = "black", fg = "white") .grid(row = 2, column = 0,padx=1,
                                             pady=1, sticky = W+E+N+S)
    enter_bkgnd_meth = Entry(param_insp, bg = "white")
    enter_bkgnd_meth.grid(row = 2, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_meth.insert(0,trackers[v].parameters['bkgnd_meth'])
    
    
    Label (param_insp,text="Number of frames in background (integer):", 
           bg = "black", fg = "white") .grid(row = 2, column = 2,padx=1,
                                             pady=1, sticky = W+E+N+S)
    enter_bkgnd_nframes = Entry(param_insp, bg = "white")
    enter_bkgnd_nframes.grid(row = 2, column = 3,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_nframes.insert(0,trackers[v].parameters['bkgnd_nframes'])
    
    
    Label (param_insp,text="Smoothing sigma (\u03bcm):", bg = "black", 
           fg = "white").grid(row = 3, column = 0,padx=1, pady=1,
           sticky = W+E+N+S)
    enter_k_sig = Entry(param_insp, bg = "white")
    enter_k_sig.grid(row = 3, column = 1,padx=1, pady=1, sticky = W+E)
    enter_k_sig.insert(0,str(trackers[v].parameters['k_sig']))
    
    
    Label (param_insp,text="Minimum frames in a track (integer):", 
           bg = "black", fg = "white") .grid(row = 3, column = 2,padx=1, 
                                             pady=1, sticky = W+E+N+S)
    enter_min_f = Entry(param_insp, bg = "white")
    enter_min_f.grid(row = 3, column = 3,padx=2, pady=1, sticky = W+E)
    enter_min_f.insert(0,trackers[v].parameters['min_f'])
    
    
    Label (param_insp,text="BW threshold (1-254):", bg = "black", 
           fg = "white") .grid(row = 4, column = 0,padx=1, pady=1, 
                               sticky = W+E+N+S)
    enter_bw_thr = Entry(param_insp, bg = "white")
    enter_bw_thr.grid(row = 4, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bw_thr.insert(0,trackers[v].parameters['bw_thr'])
    
    
    Label (param_insp,text="Distance threshold (\u03bcm):", bg = "black",
        fg = "white").grid(row = 4, column = 2,padx=1, pady=1,
        sticky = W+E+N+S)
    enter_d_thr = Entry(param_insp, bg = "white")
    enter_d_thr.grid(row = 4, column = 3,padx=1, pady=1, sticky = W+E)
    enter_d_thr.insert(0,trackers[v].parameters['d_thr'])
    
    
    Label (param_insp,text="Minimum area (\u03bcm\u00b2):", bg = "black", 
           fg = "white") .grid(row = 5, column = 0,padx=1, pady=1, 
                               sticky = W+E+N+S)
    enter_min_sz = Entry(param_insp, bg = "white")
    enter_min_sz.grid(row = 5, column = 1,padx=1, pady=1, sticky = W+E)
    enter_min_sz.insert(0,trackers[v].parameters['area_bnds'][0])
    
    
    Label (param_insp,text="Maximum area (\u03bcm\u00b2)", bg = "black", 
           fg = "white") .grid(row = 5, column = 2,padx=1, pady=1,
                               sticky = W+E+N+S)
    enter_max_sz = Entry(param_insp, bg = "white")
    enter_max_sz.grid(row = 5, column = 3,padx=1, pady=1, sticky = W+E)
    enter_max_sz.insert(0,trackers[v].parameters['area_bnds'][1])
    
    
    Label (param_insp,text="Size change threshold (%)", bg = "black",
           fg = "white") .grid(row = 6, column = 0,padx=1, pady=1, 
                               sticky = W+E+N+S)
    enter_del_sz_thr = Entry(param_insp, bg = "black")
    enter_del_sz_thr.grid(row = 6, column = 1,padx=1, pady=1, sticky = W+E)
    #enter_del_sz_thr.insert(0,trackers[v].parameters['del_sz_thr'])
    
    
    Button(param_insp,text="Scale (\u03bcm per pixel) (click for GUI)", 
           command = find_scale_button, bg = "black", fg = "white") \
        .grid(row = 6, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_um_per_pix = Entry(param_insp, bg = "white")
    enter_um_per_pix.grid(row = 6, column = 3,padx=1, pady=1, sticky = W+E)
    enter_um_per_pix.insert(0,trackers[v].parameters['um_per_pix'])
    
    
    Button(param_insp,text="mask R-CNN file (click to choose):", 
           command = choose_mrcnn_file_button, bg = "black", fg = "white") \
        .grid(row = 7, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_mrcnn_file = Entry(param_insp, bg = "white")
    enter_mrcnn_file.grid(row = 7, column = 1,padx=1, pady=1, sticky = W+E)
    enter_mrcnn_file.insert(0,trackers[v].parameters['mask_RCNN_file'])
    
    
    Button(param_insp,text="behavior model file (click to choose):", 
           command = choose_behavior_model_button, bg = "black", 
           fg = "white").grid(row = 7, column = 2,padx=1, pady=1,
           sticky = W+E+N+S)
    enter_behavior_model_file = Entry(param_insp, bg = "white")
    enter_behavior_model_file.grid(
        row = 7, column = 3,padx=1, pady=1, sticky = W+E)
    enter_behavior_model_file.insert(
        0,trackers[v].parameters['behavior_model_file'])
    
    # set up buttons
    Button(param_insp, text = "COMPUTE BACKGROUND") \
        .grid(row = 8, column = 0, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "UPDATE IMAGES",command = update_images_button)\
        .grid(row = 8, column = 1, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "CYCLE IMAGE", command = cycle_image_button) \
        .grid(row = 8, column = 2, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "SAVE AND EXIT", command = save_exit_button) \
        .grid(row = 8, column = 3, padx=1, pady=1, sticky = W+E+N+S)
    
    param_insp.mainloop()
    
    # wrapping up
    return trackers

# testing
if __name__ == '__main__':
    try:
        import sys
        sys.path.append(os.path.split(__file__)[0])
        import tracker as tracker
        vid_name = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Celegans_vid_cropped_scaled\Luca_T2_Rep1_day60002 22-01-18 11-49-24_crop_1_to_300_inc_3_scl_0.5.avi'
        tracker_obj = tracker.Tracker(vid_name)
        out = parameter_GUI([tracker_obj])
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
            

# C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Bram_vids_cropped2