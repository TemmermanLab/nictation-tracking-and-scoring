# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:52:01 2021

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

def head_tail_inspector(vid_name, vid_path, centroids, centerlines, first_frames, halfwidth, scores = -1):
    
    # define variables
    if scores == -1:
        scores = np.zeros(np.shape(centroids)[0],dtype = np.int32)
    w = 0; f = -1; flipped = False
    
    toggle_pressed = False
    censor_pressed = False
    ok_pressed = False
    exit_pressed = False
    
    # video loop function
    def play():
        nonlocal toggle_pressed, censor_pressed, ok_pressed, scores, centerlines
        print('play function running')
        Q = False
        w = 0
        w_ref = 0
        vign_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking\\vignettes'
        
        while Q is False:
            if w == len(centerlines):
                print('finished scoring')
                break
            elif Q:
                break
            
            # if w == 0 or w != w_ref:
            vign_name = 'w'+str(w)+'.avi'
            vign = cv2.VideoCapture(vign_path+'\\'+vign_name)
            fps = vign.get(cv2.CAP_PROP_FPS)
            print('scoring worm '+str(w+1)+' of '+str(len(centerlines)))
            
            # first = first_frames[w]
            # last = first_frames[w]+len(centroids[w])
            # cent_frames = np.uint32(np.linspace(first,last-1,last-first))
            #vign_f = 0
            
            for f in range(len(centroids[w])):
                
                t0 = time.time()
                vign.set(cv2.CAP_PROP_POS_FRAMES, f)
                frame = vign.read()[1]
                #vign_f = vign_f+1
                centroid = np.uint16(np.round(centroids[w][f]))
                centerline = centerlines[w][f,:,:]
                H1 = tuple(np.uint16(centerline[0][0]-centroid+halfwidth));
                H2 = tuple(np.uint16(centerline[0][-1]-centroid+halfwidth));
                #frame = frame[centroid[1]-halfwidth:centroid[1]+halfwidth,centroid[0]-halfwidth:centroid[0]+halfwidth,:]
                frame = cv2.circle(frame, H1, 2, (255,0,0), 3)
                # if w == 15:
                #     import pdb; pdb.set_trace()
                frame = Im.fromarray(frame)
                frame = frame.resize((600,600),Im.NEAREST)
                frame = ImageTk.PhotoImage(frame)
                img_win.configure(image = frame)
                img_win.update()
                
                msg2 = 'frame '+str(f+1)+' of '+str(len(centroids[w]))
                print(msg2)
                
                # slow video to recorded frame rate
                t_elap = time.time()-t0
                wait_time = (1.0/fps)-t_elap
                if wait_time > 0:
                    time.sleep(wait_time)
                
                if toggle_pressed == True:
                    if scores[w] == 0:
                        scores[w] = 2
                        centerlines[w] = np.flip(centerlines[w],2)
                        print('worm '+ str(w+1) + ' flipped!')
                    elif scores[w] == 2:
                        scores[w] = 1
                        centerlines[w] = np.flip(centerlines[w],2)
                        print('worm '+ str(w+1) + ' flipped back!')
                    elif scores[w] == 1:
                        scores[w] = 2
                        centerlines[w] = np.flip(centerlines[w],2)
                        print('worm '+ str(w+1) + ' flipped again!')
                    toggle_pressed = False
                
                elif ok_pressed == True:
                    if scores[w] == 0:
                        scores[w] = 1
                    w = w + 1
                    flipped = False
                    ok_pressed = False
                    print('worm '+str(w+0) + ' score saved')
                    break
                    
                elif censor_pressed:
                    scores[w] = 3
                    w = w + 1
                    flipped = False
                    censor_pressed = False
                    print('worm '+str(w+0) + ' censored')
                    break
                
                elif exit_pressed:
                    Q = True
                    vign.release()
                    break
        
    # button functions           
    def toggle_button():
        nonlocal toggle_pressed
        print('toggle button pressed')
        toggle_pressed = True
        
    def ok_button():
        nonlocal ok_pressed
        print('ok button pressed')
        ok_pressed = True
        
    def censor_button():
        nonlocal censor_pressed
        print('censor button pressed')
        censor_pressed = True            

    def exit_button():
        nonlocal exit_pressed
        print('exit button pressed')
        exit_pressed = True

    # set up GUI
    head_tail_insp = tk.Tk()
    head_tail_insp.title('Head/Tail Inspection GUI')
    head_tail_insp.configure(background = "black")
    
    # set up video window
    #pdb.set_trace()
    img = np.zeros((600,600),dtype = 'uint8')
    ph = ImageTk.PhotoImage(image = Im.fromarray(img))
    img_win = Label(head_tail_insp,image = ph, bg = "black", width = 600)
    #img_win.image = ph
    img_win.grid(row = 0, column = 0, columnspan = 4, padx = 2, pady = 2)
    
    # set up buttons
    Button(head_tail_insp, text = "TOGGLE", command = toggle_button) .grid(row = 1, column = 0, padx=1, pady=1, sticky = W+E+N+S)
    Button(head_tail_insp, text = "OK", command = ok_button) .grid(row = 1, column = 1, padx=1, pady=1, sticky = W+E+N+S)
    Button(head_tail_insp, text = "CENSOR", command = censor_button) .grid(row = 1, column = 2, padx=1, pady=1, sticky = W+E+N+S)
    Button(head_tail_insp, text = "EXIT", command = exit_button) .grid(row = 1, column = 3, padx=1, pady=1, sticky = W+E+N+S)
    
    #centerlines, scores = play(cap, scores)
    #head_tail_insp.wait_visibility(play(cap))
    play()
    head_tail_insp.destroy()
    head_tail_insp.quit()
    # head_tail_insp.mainloop() # not necessay, tk.update() is sufficient
    
    
    # wrapping up
    return centroids, centerlines, scores