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
    -cartoon showing worm size and distance threshold, and size threshold would help
    

@author: PDMcClanahan
"""

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb, traceback, sys, code
from scipy import interpolate
from PIL import Image as Im, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import *
import pickle
import time


def parameter_GUI(trackers):
    
    # vars for display
    bkgnd_meths = ('max_merge','min_merge')
    img_type_inds = [0,1,2,3,4,5]
    img_type_ind = 5
    f = 0
    v = 0

    # init outcome values
    bkgnd = trackers[v].background
    f_width = 600
    f_height = int(np.round(600*(np.shape(bkgnd)[0]/np.shape(bkgnd)[1])))
    img, diff, smooth, bw, bw_sz, final = trackers[v].show_segmentation(v)
    
    
    def update_images_button():
        print('Updating images...')
        nonlocal trackers, v, f
        nonlocal img, diff, smooth, bw, bw_sz, final
        
        try:
            v = int(enter_video_num.get())-1
            f = int(enter_frame_num.get())-1
            trackers[v].parameters['human_checked'] = True
            
            if enter_bkgnd_meth.get() is not trackers[v].parameters['bkgnd_meth']:
                trackers[v].parameters['bkgnd_meth'] = enter_bkgnd_meth.get()
                trackers[v].get_background()
            trackers[v].parameters['bkgnd_nframes'] = int(enter_bkgnd_nframes.get())
            trackers[v].parameters['k_sig'] = float(enter_k_sig.get())
            trackers[v].parameters['min_f'] = int(enter_min_f.get())
            trackers[v].parameters['bw_thr'] = int(enter_bw_thr.get())
            trackers[v].parameters['d_thr'] = int(enter_d_thr.get())
            trackers[v].parameters['area_bnds'] = (int(enter_min_sz.get()),int(enter_max_sz.get()))
        except:
            print('Enter numbers in all the fields')
            import pdb; pdb.set_trace()
        #trackers[v].parameters['del_sz_thr'] = int(enter_del_sz_thr.get())
        if enter_um_per_pix.get() == '' or enter_um_per_pix.get() == 'None':
            trackers[v].parameters['um_per_pix'] = 'None'
        else:
            trackers[v].parameters['um_per_pix'] = float(enter_um_per_pix.get())
        
        if v < len(trackers) and f < trackers[v].num_frames:
            try:
                img, diff, smooth, bw, bw_sz, final = trackers[v].show_segmentation(f)
            except:
                import pdb
                import sys
                import traceback
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
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
        pass        
        
    def save_exit_button():
        
        ### move this under an "apply all" button in the future
        nonlocal trackers
        for t in trackers:
            try:
                if enter_bkgnd_meth.get() is not t.parameters['bkgnd_meth']:
                    t.parameters['bkgnd_meth'] = enter_bkgnd_meth.get()
                    t.get_background()
                t.parameters['bkgnd_nframes'] = int(enter_bkgnd_nframes.get())
                t.parameters['k_sig'] = float(enter_k_sig.get())
                t.parameters['min_f'] = int(enter_min_f.get())
                t.parameters['bw_thr'] = int(enter_bw_thr.get())
                t.parameters['d_thr'] = int(enter_d_thr.get())
                t.parameters['area_bnds'] = (int(enter_min_sz.get()),int(enter_max_sz.get()))
            except:
                # import pdb; pdb.set_trace()
                print('Enter numbers in all the fields')
            #trackers[v].parameters['del_sz_thr'] = int(enter_del_sz_thr.get())
            if enter_um_per_pix.get() == '':
                t.parameters['um_per_pix'] = 'None'
            else:
                try:
                    t.parameters['um_per_pix'] = float(enter_um_per_pix.get())
                except:
                    t.parameters['um_per_pix'] = enter_um_per_pix.get()
        ###
        
        param_insp.destroy()
        param_insp.quit()
    
    def update_win(img_type_ind):
        nonlocal frame
        # pdb.set_trace()
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
    
    # def get_scale():
    #     # get and load scale image
    #     root = tk.Tk()
    #         initialdir = os.path.dirname(vid_name)
    #         scale_img_file = tk.filedialog.askopenfile(initialdir = '/', \
    #             title = "Select the folder containing videos to be tracked \
    #             ...")
    #         root.destroy()
    #     img = cv2.imread(scale_img_file,cv2.IMREAD_GRAYSCALE)
    #     imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    #     # display image
    #     clicks_x,clicks_y = [],[]
        
        
    #     # measure an object in the scale image
    #     return um_per_pix
        
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
    Label (param_insp,text="Video:", bg = "gray", fg = "black") .grid(row = 1, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_video_num = Entry(param_insp, bg = "white")
    enter_video_num.grid(row = 1, column = 1,padx=1, pady=1, sticky = W+E)
    enter_video_num.insert(0,str(v+1))
    
    
    Label (param_insp,text="Frame:", bg = "gray", fg = "black") .grid(row = 1, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_frame_num = Entry(param_insp, bg = "white")
    enter_frame_num.grid(row = 1, column = 3,padx=1, pady=1, sticky = W+E)
    enter_frame_num.insert(0,str(f+1))
    
    
    Label (param_insp,text="Background method (max_merge or min_merge):", bg = "black", fg = "white") .grid(row = 2, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_bkgnd_meth = Entry(param_insp, bg = "white")
    enter_bkgnd_meth.grid(row = 2, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_meth.insert(0,trackers[v].parameters['bkgnd_meth'])
    
    
    Label (param_insp,text="Number of frames in background (integer):", bg = "black", fg = "white") .grid(row = 2, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_bkgnd_nframes = Entry(param_insp, bg = "white")
    enter_bkgnd_nframes.grid(row = 2, column = 3,padx=1, pady=1, sticky = W+E)
    enter_bkgnd_nframes.insert(0,trackers[v].parameters['bkgnd_nframes'])
    
    
    Label (param_insp,text="Smoothing sigma:", bg = "black", fg = "white") .grid(row = 3, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_k_sig = Entry(param_insp, bg = "white")
    enter_k_sig.grid(row = 3, column = 1,padx=1, pady=1, sticky = W+E)
    enter_k_sig.insert(0,str(trackers[v].parameters['k_sig']))
    
    
    Label (param_insp,text="Minimum frames in a track (integer):", bg = "black", fg = "white") .grid(row = 3, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_min_f = Entry(param_insp, bg = "white")
    enter_min_f.grid(row = 3, column = 3,padx=2, pady=1, sticky = W+E)
    enter_min_f.insert(0,trackers[v].parameters['min_f'])
    
    
    Label (param_insp,text="BW threshold (1-254):", bg = "black", fg = "white") .grid(row = 4, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_bw_thr = Entry(param_insp, bg = "white")
    enter_bw_thr.grid(row = 4, column = 1,padx=1, pady=1, sticky = W+E)
    enter_bw_thr.insert(0,trackers[v].parameters['bw_thr'])
    
    
    Label (param_insp,text="Distance threshold:", bg = "black", fg = "white") .grid(row = 4, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_d_thr = Entry(param_insp, bg = "white")
    enter_d_thr.grid(row = 4, column = 3,padx=1, pady=1, sticky = W+E)
    enter_d_thr.insert(0,trackers[v].parameters['d_thr'])
    
    
    Label (param_insp,text="Minimum area (integer):", bg = "black", fg = "white") .grid(row = 5, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_min_sz = Entry(param_insp, bg = "white")
    enter_min_sz.grid(row = 5, column = 1,padx=1, pady=1, sticky = W+E)
    enter_min_sz.insert(0,trackers[v].parameters['area_bnds'][0])
    
    
    Label (param_insp,text="Maximum area (integer)", bg = "black", fg = "white") .grid(row = 5, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_max_sz = Entry(param_insp, bg = "white")
    enter_max_sz.grid(row = 5, column = 3,padx=1, pady=1, sticky = W+E)
    enter_max_sz.insert(0,trackers[v].parameters['area_bnds'][1])
    
    
    Label (param_insp,text="Size change threshold (percentage)", bg = "black", fg = "white") .grid(row = 6, column = 0,padx=1, pady=1, sticky = W+E+N+S)
    enter_del_sz_thr = Entry(param_insp, bg = "black")
    enter_del_sz_thr.grid(row = 6, column = 1,padx=1, pady=1, sticky = W+E)
    #enter_del_sz_thr.insert(0,trackers[v].parameters['del_sz_thr'])
    
    
    Button(param_insp,text="Scale in \u03bcm per pixel (float)", command = find_scale_button, bg = "black", fg = "white") .grid(row = 6, column = 2,padx=1, pady=1, sticky = W+E+N+S)
    enter_um_per_pix = Entry(param_insp, bg = "white")
    enter_um_per_pix.grid(row = 6, column = 3,padx=1, pady=1, sticky = W+E)
    enter_um_per_pix.insert(0,trackers[v].parameters['um_per_pix'])
    
    # set up buttons
    Button(param_insp, text = "COMPUTE BACKGROUND") .grid(row = 7, column = 0, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "UPDATE IMAGES", command = update_images_button) .grid(row = 7, column = 1, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "CYCLE IMAGE", command = cycle_image_button) .grid(row = 7, column = 2, padx=1, pady=1, sticky = W+E+N+S)
    Button(param_insp, text = "SAVE AND EXIT", command = save_exit_button) .grid(row = 7, column = 3, padx=1, pady=1, sticky = W+E+N+S)
    
    param_insp.mainloop()
    
    # wrapping up
    return trackers

# testing
if __name__ == '__main__':
    try:
        import sys
        sys.path.append(os.path.split(__file__)[0])
        import tracker_classes as tracker
        vid_name = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Bram_vids_cropped2\video_AX7163_A 21-09-23 15-37-15_crop_1_to_300_inc_3.avi'
        tracker_obj = tracker.Tracker(vid_name)
        bkgnd_meth, bkgnd_nframes, k_sig, k_sz, bw_thr, sz_bnds, d_thr, \
            del_sz_thr, um_per_pix, min_f = parameter_GUI([tracker_obj])
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
            

# C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Bram_vids_cropped2