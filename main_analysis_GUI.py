# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:13:11 2021

This GUI allows click-through execution of the worm tracking code. The
workflow is:
    
    1. Select videos to track (already-selected videos are displayed in the 
       GUI window)
    2. Set tracking parameters (opens an instance of the parameter GUI)
    3. Run tracking, calculate features, and scoring (option to do all three
       without further input)
    4. Exit

Known issues and improvements:
    -option to pick up where left off for partially-done analysis, useful if
     computer crashes during a large chunk
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
import tkinter.filedialog as filedialog # necessary to avoid error
from tkinter.messagebox import askyesno
import numpy as np
import os
import time

import sys
sys.path.append(os.path.split(__file__)[0])

import parameter_GUI
import tracker as tracker


# combining torch with matplotlib or numpy's linalg.lstsq kills the kernel w/o
# this
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# hard-coded settings (for now)
multiprocessing = False
if multiprocessing:
    from multiprocess import Pool
    #from pathos.multiprocessing import ProcessPool

    

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
        

    
    # load a video or videos to be analyzed
    def load_folder():
        nonlocal trackers, data_path
        
        root = tk.Tk()
        data_path = tk.filedialog.askdirectory(initialdir = '/', \
            title = "Select the folder containing videos to be analyzed \
            ...")
        root.destroy()
        
        print('Fetching video info '+data_path)
        
        vid_names = sorted(os.listdir(data_path))
        for v in reversed(range(len(vid_names))):
            if len(vid_names[v])<4 or vid_names[v][-4:] != '.avi':
                pass
            else:
                trackers.append(tracker.Tracker(data_path+'//'+vid_names[v]))
                
        update_vid_inf(trackers)

    
    # button functions
    
    def load_video_folder_button():
        load_folder()
        
        
    def set_parameters_button():
        nonlocal trackers
        trackers = parameter_GUI.parameter_GUI(trackers)
        for t in trackers:
            t.save_params()
        update_vid_inf(trackers)

    
    def track_button():
        nonlocal trackers # was not needed before adding pooling, ikd why
        
        # ask if user wants to also calulate features and score behavior
        root = tk.Tk()
        keep_going = askyesno(title='option to run full analysis now',
         message='Do you also want to calculate features and score behavior?')
        root.destroy()
        
        t0 = time.time()
        
        #import pdb; pdb.set_trace()
        if multiprocessing:
            print('Using multiprocessing')
            
 
            # # this does nothing at all
            # def run_analysis(t):
            #     t.track()
            #     #if keep_going:
            #     t.calculate_features()
            #     t.score_behavior()
            #     a = 1
            #     return a
            
            # import pdb; pdb.set_trace()
            # p = Pool(3)
            # ys_pool = p.map_async(run_analysis, trackers)
            
            def run_analysis(vid_file):
                
                # re-initialize the tracker objects here in the pool function
                dummy = 1
                t = tracker.Tracker(vid_file)
                t.track()
                t.calculate_features()
                t.score_behavior()
                t.behavior_summary_video()
                return dummy
            
            v_files = []
            for t in trackers:
                v_files.append(t.vid_path+'//'+t.vid_name)
                # keep_goings.append(keep_going)
            import pdb; pdb.set_trace()
            del trackers # not sure if this is necessary 
            p = Pool(3)
            ys_pool = p.map_async(run_analysis, v_files)
            
            ##################################################################
            
            # # fails, does not do anything when run in Spyder
            # def run_analysis(vid_file, keep_going):
            #     # re-initialize the tracker objects here in the pool function
            #     import pdb; pdb.set_trace()
            #     dummy = 1
            #     t = tracker.Tracker(vid_file)
            #     t.track()
            #     if keep_going:
            #         t.calculate_features()
            #         t.score_behavior()
            #     return dummy
            
            # def run_analysis(vid_file, keep_going):
            #     return 4
            
            # # you cannot pass non picklable things like cv2.VideoCapture
            # # objects to the pool function, so the workaround is to 
            # # re-initialize the tracker objects inside the pool function
            # # run_analysis
            # v_files = []; keep_goings = []
            
            # for t in trackers:
            #     v_files.append(t.vid_path+'//'+t.vid_name)
            #     keep_goings.append(keep_going)
            
            # del trackers # not sure if this is necessary 
            
            # pool = ProcessPool(nodes=3)
            # dummies = pool.map(run_analysis, v_files, keep_goings) # non blocking map, seems fastest
            
            ##################################################################
            
            # # fails, something about pickling
            # def run_analysis(tracker, keep_going):
            #     t.track()
            #     if keep_going:
            #         t.calculate_features()
            #         t.score_behavior()
            
            # jobs = []
            
            # for t in trackers:
            #     #out_list = list()
            #     process = multiprocessing.Process(target=run_analysis, 
            #                                       args=(t, keep_going))
            #     jobs.append(process)
        
            # # Start the processes (i.e. calculate the random number lists)      
            # for j in jobs:
            #     j.start()
        
            # # Ensure all of the processes have finished
            # for j in jobs:
            #     j.join()
            
            
        else:
            
            for t in trackers:
                t.track()
                
                if keep_going:
                    t.calculate_features()
                    t.score_behavior()
                    t.behavior_summary_video()
        
        print('WARNING: tracking objects deleted')
            #update_vid_inf(trackers)
        
        t_elap = time.time()-t0 
        
        if t_elap < 60:
            print('Tracking took ' + str(round(t_elap,1)) + \
                   ' seconds for ' + str(len(trackers)) + ' videos.')
        elif t_elap < 3600:
            print('Tracking took ' + str(round(t_elap/60,1)) + \
                   ' minutes for ' + str(len(trackers)) + ' videos.')
        else:
            print('Tracking took ' + str(round(t_elap/3600,1)) + \
                   ' hours for ' + str(len(trackers)) + ' videos.')
                
        print('Finished ' + str(time.ctime()))


    def calculate_features_button():
        for t in trackers:
            try:
                t.calculate_features()
                print('Done calculating features!')

            except:
                import pdb
                import sys
                import traceback
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
            
    
    def score_button():
        for t in trackers:
            t.score_behavior()
            t.behavior_summary_video()
         
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
    tracking_GUI.title('Main Analysis GUI')
    tracking_GUI.configure(background = "black")
    # get character size / line spacing in pixels
    chr_h_px = tkinter.font.Font(
        root = tracking_GUI, font=('Courier',12,'normal')).metrics('linespace')
    chr_w_px = tkinter.font.Font(
        root = tracking_GUI, font=('Courier',12,'normal')).measure('m')
    
    # make the main window as wide and a bit taller than the vid info window
    tracking_GUI.geometry(str(int(w*chr_w_px))+"x"+str(int(chr_h_px*(h+3))))
    
    
    # to do text
    todo_txt = tk.Label(text = 'load a folder containing videos for analysis')
    todo_txt.grid(row = 0, column = 0, columnspan = 6, padx = 0, pady = 0)


    # informational window
    vid_inf = tkinter.Text(tracking_GUI, height = h, width = w)
    vid_inf.configure(font=("Courier", 12))
    vid_inf.grid(row = 1, column = 0, columnspan = 6, padx = 0, pady = 0)
    
    
    # buttons
    tk.Button(tracking_GUI,
              text = "LOAD VIDEO FOLDER",
              command = load_video_folder_button,
              width = 10) \
              .grid(row = 2,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(tracking_GUI,
              text = "SET PARAMETERS",
              command = set_parameters_button,
              width = 10) \
              .grid(row = 2,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(tracking_GUI,
              text = "TRACK",
              command = track_button,
              width = 10) \
              .grid(row = 2,
                    column = 2,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(tracking_GUI,
              text = "CALCULATE FEATURES",
              command = calculate_features_button,
              width = 10) \
              .grid(row = 2,
                    column = 3,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(tracking_GUI,
              text = "SCORE",
              command = score_button,
              width = 10) \
              .grid(row = 2,
                    column = 4,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(tracking_GUI,
              text = "EXIT",
              command = exit_button,width = 10) \
              .grid(row = 2,
                    column = 5,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    

    tracking_GUI.mainloop()


if __name__ == '__main__':
    # needed for multiprocess
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    try:
        tracking_GUI()
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    






