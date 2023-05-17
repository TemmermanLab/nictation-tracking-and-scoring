# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:55:45 2023

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
    
def cross_validation_GUI():
    
    train_dir = []
    test_dir = []
    algorithm = []
    k = 5
    
    xval_GUI = tk.Tk()
    
    train_label = Label(xval_GUI,text="Training dataset:")
    train_label.grid(row=0, column=0, columnspan = 1,sticky="w")
    train_entry = Entry(xval_GUI)
    train_entry.grid(row=0, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    val_label = Label(xval_GUI,text="Validation dataset:")
    val_label.grid(row=1, column=0, columnspan = 1,sticky="w")
    val_entry = Entry(xval_GUI,text="")
    val_entry.grid(row=1, column=1, columnspan = 3,sticky = 'W'+'E'+'N'+'S')
    
    
    scale_meth_label = Label(xval_GUI,text="Scaling method:")
    scale_meth_label.grid(row=2, column=0, columnspan = 1,sticky="w")
    scaling_methods = ['none','min max','variance','Gaussian','whiten']
    scale_meth_strv = StringVar()
    scale_meth_strv.set( "none" )
    scale_meth_drop = OptionMenu(xval_GUI , scale_meth_strv , *scaling_methods) \
    .grid(row = 2,column = 1,columnspan=3,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
    alg_label = Label(xval_GUI,text="Algorithm:")
    alg_label.grid(row=3, column=0, columnspan = 1,sticky="w")
    algorithms =['logistic regression','decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes',
        'support vector machine', 'random forest', 'neural network']
    alg_strv = StringVar()
    alg_strv.set( "random forest" )
    algorithm_drop = OptionMenu(xval_GUI , alg_strv , *algorithms) \
    .grid(row = 3,column = 1,columnspan=3,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
        
    def select_training_set_button():
        root = tk.Tk()
        train_dir = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing features and manual scores for training... \
                ...")
        root.destroy()
        train_entry.delete(0, tk.END)
        train_entry.insert(0, train_dir)
        
    
    def select_validation_set_button():
        root = tk.Tk()
        val_dir = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing features and manual scores for validation... \
                ...")
        root.destroy()
        val_entry.delete(0, tk.END)
        val_entry.insert(0, val_dir)
        
    
    def run_button():
        train_dir = train_entry.get()
        val_dir = val_entry.get()
        scale_meth = scale_meth_strv.get()
        alg = alg_strv.get()
        df = \
            nm.k_fold_cross_validation(train_dir, alg, scale_meth, val_dir,
                                          k)
        df.rename(columns = {'val acc':'val acc ('+val_dir+')'}, inplace = True)
        df.to_csv(f"{train_dir}//{alg}_{scale_meth}_{k}_fold_cross_validation_accuracies.csv",index = False)                     
        
    def exit_button():
        xval_GUI.destroy()
        xval_GUI.quit()
    
    
    
    tk.Button(xval_GUI,
              text = "SELECT TRAINING SET",
              command = select_training_set_button) \
              .grid(row = 4,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(xval_GUI,
              text = "SELECT VALIDATION SET",
              command = select_validation_set_button) \
              .grid(row = 4,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    tk.Button(xval_GUI,
              text = "RUN",
              command = run_button) \
              .grid(row = 4,
                    column = 2,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(xval_GUI,
              text = "EXIT",
              command = exit_button) \
              .grid(row = 4,
                    column = 3,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    xval_GUI.title('Cross Validation GUI')
    
    curr_row = 0
    
    mainloop()


if __name__ == '__main__':
    
    try:
        cross_validation_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)