# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:15 2021

# This version scores the following categories:
    1. Censored (-1)
    2. Quiescent + Recubent (0)
    3. Active + Recumbent (1)
    4. Active + Nictating (2)
    5. Quiescent + Nictating (3)

# issues and improvements:
-make it so that indexing is consistent (right now the GUI shows tracks as 1-
 indexed, but the spreadsheet is 0-indexed
-change spreadsheet headers to actual vignette names rather than 'track_0' etc.
-add a button to rewind to beginning and re-score a track
-might be easier with play / pause, step on either side, and separate speed 
 control
-do not record a score if the video was never advanced (right now if you go 
 forward into the unscored track and backward again, the first frame is given
 the -1 censored score)
-censor track button
-auto-saving

@author: PDMcClanahan
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label
from tkinter import *
from PIL import ImageTk, Image
import os
import time
import pickle
import copy
import csv
import pdb
from functools import partial

def nictation_scoring_GUI(vignette_path = 'null'):
    

    
    # video operation and scoring functions
    
    def play():
        nonlocal f, play_state
        global vign
        print('play function running')
        fps = vign.get(cv2.CAP_PROP_FPS)
        
        #while f < 50:
        while play_state > 0:
            
            #if play_state == 1:
            t0 = time.time()
            
            update_scores()
            if f < len(scores[w])-1:
                f = f+1
                update_still()
            else:
                play_state = 0
            
            t_elap = time.time()-t0
            if play_state == 1:
                wait_time = (1.0/fps)-t_elap
                if wait_time > 0:
                    time.sleep(wait_time)
                else: print('WARNING: video playing slower than real time')
            else:
                wait_time = (1.0/(fps*3))-t_elap
                if wait_time > 0:
                    time.sleep(wait_time)
         
    def pause():
        nonlocal f, play_state
        
        if play_state == 2: # rewind slightly if fast-forwarding
            f = f - 4
        
        play_state = 0
        update_still()
   
    
    def update_still():
        nonlocal img_win
        global frame
        #global img_win
        vign.set(cv2.CAP_PROP_POS_FRAMES, f)
        frame = vign.read()[1]
        frame = cv2.putText(frame, score_descriptions[score+1],
                            tuple(text_origin), font, font_scale,
                            tuple(text_colors[score+1]), text_thickness,
                            cv2.LINE_AA)
        frame = cv2.putText(frame, 'frame '+str(f+1)+'/'+str(len(scores[w])),
                            tuple(text_origin_2), font, font_scale, (16,78,139),
                            text_thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, 'track '+str(w+1)+'/'+str(len(scores)),
                            tuple(text_origin_3), font, font_scale, (16,78,139),
                            text_thickness, cv2.LINE_AA)
        frame = Image.fromarray(frame)
        frame = frame.resize((600,600),Image.NEAREST) 
        frame = ImageTk.PhotoImage(frame)
        img_win.configure(image = frame)
        img_win.update()
 
    
    def update_scores():
        nonlocal scores
        scores[w][f] = score

    
    def update_vign():
        global vign, vign_num_f
        if 'vign' in globals():
            vign.release()
        vign = cv2.VideoCapture(vignette_path+'/'+vignette_list[w])
        vign_num_f = vign.get(cv2.CAP_PROP_FRAME_COUNT)           

    
    def save_scores_pickle():
        pickle.dump(scores, open(save_file_pickle,'wb'))
        print('scores saved in '+ save_file_pickle)

    
    def save_scores_csv():
        nonlocal vignette_list
        
        with open(save_file_csv, mode='w',newline="") as csv_file:
            #pdb.set_trace()
            scores_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            # complicated because csv writer only writes rows
            row = []
            for ww in range(len(scores)): row.append(vignette_list[ww][0:-4])
            scores_writer.writerow(row)
            
            num_frames = []
            for s in scores: num_frames.append(len(s))
            num_r = np.max(num_frames)
            for r in range(num_r):
                row = []
                for ww in range(len(scores)):
                    if r < len(scores[ww]):
                        row.append(scores[ww][r])
                    else:
                        row.append('')
                scores_writer.writerow(row)
                
                
                
    # button functions
    def toggle_score_button(fwd = True):
        nonlocal score
        if fwd:
            if score < (len(score_descriptions)-2):
                score = score + 1
            else:
                score = -1
        else:
            if score > -1:
                score = score - 1
            else:
                score = len(score_descriptions)-2
        if play_state == 0:
            update_still()

        
    def step_backward_button():
        nonlocal play_state, f, score, scores
        print('step backward button pressed')
        #if play_state != 0:
        if f > 0:
            if play_state == 2: # rewind slightly if fast-forwarding
                play_state = 0
                f = f - 4
            else:
                play_state = 0
                f = f - 1
            score = scores[w][f]
            update_still()

        
    def pause_button():
        print('pause button pressed')
        pause()

    
    def step_forward_button():
        nonlocal play_state, f
        print('step forward button pressed')
        
        if f < len(scores[w])-1:
            play_state = 0
            update_scores()
            f = f + 1
            update_still()

    
    def play_button():
        nonlocal play_state
        print('play button pressed')
        play_state = 1
        play()

        
    def fast_forward_button():
        nonlocal play_state
        print('fast forward button pressed')
        play_state = 2
        play()

        
    def previous_track_button():
        nonlocal play_state, w, f, score
        print('previous track button pressed')
        play_state = 0
        if w > 0:
            w = w - 1
            update_vign()
            if -2 in scores[w]:
                f = np.min(np.where(scores[w]==-2))
            else:
                f = int(vign_num_f)-1
                print('track ' + str(w) + \
                      ' already scored, re-score or switch tracks')
            
            score = scores[w][f]
            if score == -2:
                score = -1
            
            update_still()

    
    def next_track_button():
        nonlocal play_state, w, f, score
        if w != -1:
            print('next track button pressed')
            if f == len(scores[w])-1:
                update_scores() # save final score
        else:
            pass
            #import pdb; pdb.set_trace()
        play_state = 0
        
        if w < (len(scores)-1):
            w = w + 1
            
            update_vign()
            
            if -2 in scores[w]:
                f = np.min(np.where(scores[w]==-2))
            else:
                f = int(vign_num_f)-1
                print('track ' + str(w) + \
                      ' already scored, re-score or switch tracks')
            
            score = scores[w][f]
            if score == -2:
                score = -1
            
            update_still()

        
    def save_scores_button():
        nonlocal play_state
        print('save scores button pressed')
        
        if f == len(scores[w])-1:
            update_scores() # save final score
        
        pause()
        save_scores_pickle()
        save_scores_csv()


    def load_batch_button(vignette_path):
        
        if vignette_path == 'null':
            root = tk.Tk()
            vignette_path = filedialog.askdirectory(initialdir = '/', \
                title = "Select the folder containing video \
                vignettes of individual tracks...")
            root.destroy()
        
        print('loading video batch in '+vignette_path)
        
        vignette_list_unsorted = os.listdir(vignette_path)
        
        # remove non-video files, if any
        for v in reversed(range(len(vignette_list_unsorted))):
            if len(vignette_list_unsorted[v])<4 or \
                vignette_list_unsorted[v][-4:] != '.avi':
                del(vignette_list_unsorted[v])
        
        # # put in list in natural order
        # digit_list = []
        # for v in vignette_list_unsorted:
        #     dig = ''
        #     for c in v:
        #         if c.isdigit():
        #             dig += c
        #     digit_list.append(int(dig))
        
        # vignette_list = [
        #     x for _,x in sorted(zip(digit_list,vignette_list_unsorted))]
        
        vignette_list = sorted(vignette_list_unsorted)
        
                
        save_file_pickle = os.path.dirname(vignette_path) + \
            '\manual_nictation_scores.p'
        save_file_csv = os.path.dirname(vignette_path) + \
            '\manual_nictation_scores.csv'
        
        if os.path.exists(save_file_pickle):
            scores = pickle.load( open(os.path.dirname(vignette_path)+ \
                                       '/manual_nictation_scores.p','rb'))
            # future check for consistency btwn scores and videos will go here
        else: 
            scores = list()
            for w in range(len(vignette_list)):
                print('getting video information for vignette '+str(w+1)+ \
                      ' of '+ str(len(vignette_list)))
                vignette = cv2.VideoCapture(vignette_path+'/' + \
                                            vignette_list[w])
                scores.append(np.array(-2*np.ones(int(
                    vignette.get(cv2.CAP_PROP_FRAME_COUNT))),dtype='int8'))
                vignette.release()
        
        return vignette_path, vignette_list, scores, save_file_pickle, \
            save_file_csv

               
    def exit_button():
        nonlocal nictation_GUI
        print('exit button pressed')
        pause()
        
        if f == len(scores[w])-1:
            update_scores() # save final score
        
        save_scores_pickle()
        save_scores_csv()
        nictation_GUI.destroy()
        nictation_GUI.quit()

   
    
    # set up
    
    # GUI
    nictation_GUI = tk.Tk()
    nictation_GUI.title('Nictation Scoring GUI')
    nictation_GUI.configure(background = "black")
    
    
    # video window
    img = np.zeros((600,600),dtype = 'uint8')
    ph = ImageTk.PhotoImage(image = Image.fromarray(img))
    img_win = Label(nictation_GUI,image = ph, bg = "black", width = 600)
    img_win.grid(row = 0, column = 0, columnspan = 6, padx = 2, pady = 2)
    


    # video control and scoring buttons
    button_width = 11
    Button(nictation_GUI, text = "TOGGLE (A)", command = toggle_score_button,
           width = button_width) .grid(row = 1, column = 0, padx=1, pady=1,
                             sticky = W+E+N+S)
    Button(nictation_GUI, text = "TOGGLE (S)", command = partial(toggle_score_button, False),
           width = button_width) .grid(row = 1, column = 1, padx=1, pady=1,
                             sticky = W+E+N+S)
    Button(nictation_GUI, text = "< (←)", command = step_backward_button,
           width = button_width) .grid(row = 1, column = 2, padx=1, pady=1,
                             sticky = W+E+N+S)
    Button(nictation_GUI, text = "> (→)", command = step_forward_button,
           width = button_width) .grid(row = 1, column = 3, padx=1, pady=1,
                             sticky = W+E+N+S)
    Button(nictation_GUI, text = ">>> (↑)", command = fast_forward_button,
           width = button_width) .grid(row = 1, column = 4, padx=1, pady=1, \
                             sticky = W+E+N+S)
    Button(nictation_GUI, text = "|| (↓)", command = pause_button,width = 11) \
        .grid(row = 1, column = 5, padx=1, pady=1, sticky = W+E+N+S)
    
    # buttons for switch videos, saving scores, etc
    Button(nictation_GUI, text = "PREVIOUS TRACK (D)", 
           command = previous_track_button) .grid(row = 2, column = 0,
           columnspan = 2, padx=1, pady=1, sticky = W+E+N+S)
    Button(nictation_GUI, text = "NEXT TRACK (F)", command = next_track_button) \
        .grid(row = 2, column = 2, columnspan = 2, padx=1, pady=1,
              sticky = W+E+N+S)
    Button(nictation_GUI, text = "SAVE SCORES", 
           command = save_scores_button) .grid(row = 2, column = 4, padx=1,
                                               pady=1, sticky = W+E+N+S)
    Button(nictation_GUI, text = "EXIT", command = exit_button) .grid(row = 2,
                                column = 5, padx=1, pady=1, sticky = W+E+N+S)
    
    
    # key binding functions
    def key_pressed(event):
        # print('blah')
        # print(event.char)
        if event.char == 'a':
            toggle_score_button(False)
        elif event.char == 's':
            toggle_score_button()
        elif event.char == 'd':
            previous_track_button()
        elif event.char == 'f':
            next_track_button()
        elif event.keysym == 'Left':
            step_backward_button()
        elif event.keysym == 'Right':
            step_forward_button()
        elif event.keysym == 'Up':
            fast_forward_button()
        elif event.keysym == 'Down':
            pause_button()
        else:
            pass
            #print(event.keysym)
            
    
    nictation_GUI.bind("<Key>",key_pressed)
    
    
    # load vignettes and scores if provided in call and run GUI
    vignette_path, vignette_list, scores, save_file_pickle, save_file_csv \
        = load_batch_button(vignette_path)
    
    # settings for text over images
    font = cv2.FONT_HERSHEY_SIMPLEX 
    text_origin = [3, 8]
    line_spacing = 11
    text_origin_2 = copy.copy(text_origin)
    text_origin_2[1] =  text_origin_2[1] + line_spacing
    text_origin_3 = copy.copy(text_origin)
    text_origin_3[1] =  text_origin_3[1] + 2*line_spacing
    font_scale = 0.3
    text_colors = [(50, 50, 50),(0, 0, 255),(255, 0, 0)]
    text_thickness = 1
    
    # initialize variables

    # find first unscored track
    for w in range(len(scores)):
        if -2 in scores[w]:
            w = w-1
            break

    
    f = 0
    score = -1 
    score_descriptions = ('censored', 'recumbent', 'nictating')
    play_state = 0 # reverse, pause, play, fast forward
    update_vign() # 
    next_track_button()
    
    
    
    # run GUI
    nictation_GUI.mainloop()
    

    

# testing
if __name__ == '__main__':
    
    import traceback
    import pdb
    
    try:
        # test_path = r'C:\Users\Temmerman Lab\Desktop\Celegans_nictation_'+\
        #     r'dataset\Ce_R2_d21_tracking\vignettes'
        # test_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\2'+\
        #     r'0220404_test_vignettes'
        # nictation_scoring_GUI(test_path)
        
        root = tk.Tk()
        vig_path = tk.filedialog.askdirectory(initialdir = '/', \
            title = "Select the folder containing videos to be scored \
            ...")
        root.destroy()
        
        nictation_scoring_GUI(vig_path)
    
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        

