# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:22:58 2023

@author: PDMcClanahan
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
import copy
import pandas as pd

from datetime import datetime


sys.path.append(os.path.split(__file__)[0])
import nictation_module as nm



def write_summary_csv():
    dir_train = train_entry.get()
    dir_val = val_entry.get()
    algorithm = algo_menu.get()
    
    evaluate_models_x_fold_cross_val(dir_train, dir_val, \
                                     model_types = algorithm)
    
def smoothing_optimization_GUI():
    
    
    model_file = []
    data_dir = []
    sig_max = 1.0
    sig_inc = 0.02
   
    
    smooth_GUI = tk.Tk()
    
    model_label = Label(smooth_GUI,text="Behavior model:")
    model_label.grid(row=0, column=0, columnspan = 1,sticky="w")
    model_entry = Entry(smooth_GUI)
    model_entry.grid(row=0, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    data_label = Label(smooth_GUI,text="Behavior dataset:")
    data_label.grid(row=1, column=0, columnspan = 1,sticky="w")
    data_entry = Entry(smooth_GUI,text="")
    data_entry.grid(row=1, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    
    sig_max_label = Label(smooth_GUI,text="Max \u03C3 (s):")
    sig_max_label.grid(row=2, column=0, columnspan = 1,sticky="w")
    sig_max_entry = Entry(smooth_GUI,text="")
    sig_max_entry.grid(row=2, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    sig_inc_label = Label(smooth_GUI,text="\u03C3 increment (s):")
    sig_inc_label.grid(row=3, column=0, columnspan = 1,sticky="w")
    sig_inc_entry = Entry(smooth_GUI,text="")
    sig_inc_entry.grid(row=3, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
        
    def select_behavior_model_button():
        root = tk.Tk()
        model_file = tk.filedialog.askopenfilename(initialdir = '/', \
                title = "Select a behavior classifier model file... \
                ...")
        root.destroy()
        model_entry.delete(0, tk.END)
        model_entry.insert(0, model_file)
        
    
    def select_dataset_button():
        root = tk.Tk()
        data_dir = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing manually scored behavior data....")
        root.destroy()
        data_entry.delete(0, tk.END)
        data_entry.insert(0, data_dir)
        
    
    def run_button():
        data_dir = data_entry.get()
        model_file = model_entry.get()
        sig_max = sig_max_entry.get()
        sig_inc = sig_inc_entry.get()
        df = nm.test_smoothing(data_dir, model_file, sig_max, sig_inc)
        
        df.rename(columns = {'val acc':'val acc ('+val_dir+')'}, inplace = True)
        df.to_csv(f"{train_dir}//{alg}_{scale_meth}_{k}_fold_cross_validation_accuracies.csv",index = False)                     
        
    def exit_button():
        smooth_GUI.destroy()
        smooth_GUI.quit()
    
    
    
    tk.Button(smooth_GUI,
              text = "SELECT BEHAVIOR MODEL FILE",
              command = select_behavior_model_button) \
              .grid(row = 4,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(smooth_GUI,
              text = "SELECT BEHAVIOR DATASET",
              command = select_dataset_button) \
              .grid(row = 4,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    tk.Button(smooth_GUI,
              text = "RUN",
              command = run_button) \
              .grid(row = 4,
                    column = 2,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(smooth_GUI,
              text = "EXIT",
              command = exit_button) \
              .grid(row = 4,
                    column = 3,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    smooth_GUI.title('Smoothing Optimization GUI')
    

    mainloop()


if __name__ == '__main__':
    
    try:
        smoothing_optimization_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)