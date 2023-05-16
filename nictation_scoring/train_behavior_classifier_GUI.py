# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:37:50 2023

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
import time

from datetime import datetime


sys.path.append(os.path.split(__file__)[0])
import nictation_module as nm


def train_behavior_classifier_GUI():

    train_classifier_GUI = tk.Tk()
    
    train_label = Label(train_classifier_GUI,text="Training dataset:")
    train_label.grid(row=0, column=0, columnspan = 1,sticky="w")
    train_entry = Entry(train_classifier_GUI)
    train_entry.grid(row=0, column=1, columnspan = 2,sticky = 'W'+'E'+'N'+'S')
    
    scale_meth_label = Label(train_classifier_GUI,text="Scaling method:")
    scale_meth_label.grid(row=1, column=0, columnspan = 1,sticky="w")
    scaling_methods = ['none','min max','variance','Gaussian','whiten']
    scale_meth_strv = StringVar()
    scale_meth_strv.set( "none" )
    scale_meth_drop = OptionMenu(train_classifier_GUI , scale_meth_strv , *scaling_methods) \
    .grid(row = 1,column = 1,columnspan=2,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
    alg_label = Label(train_classifier_GUI,text="Algorithm:")
    alg_label.grid(row=2, column=0, columnspan = 1,sticky="w")
    algorithms =['logistic regression','decision tree', 'k nearest neighbors',
        'linear discriminant analysis', 'Gaussian naive Bayes',
        'support vector machine', 'random forest', 'neural network']
    alg_strv = StringVar()
    alg_strv.set( "random forest" )
    algorithm_drop = OptionMenu(train_classifier_GUI , alg_strv , *algorithms) \
    .grid(row = 2,column = 1,columnspan=2,padx=0,pady=0,sticky = 'W'+'E'+'N'+'S') 
    
        
    def select_training_set_button():
        root = tk.Tk()
        train_dir = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select a folder containing features and manual scores for training... \
                ...")
        root.destroy()
        train_entry.delete(0, tk.END)
        train_entry.insert(0, train_dir)
               
    
    def run_button():
        train_dir = train_entry.get()
        scale_meth = scale_meth_strv.get()
        alg = alg_strv.get()
        
        print('Training model...')
        model, scaler = nm.train_behavior_classifier(train_dir,
                                           scale_meth, alg)
                
        model_file = train_dir+'\\'+datetime.now().strftime("%Y%m%d%H%M%S")+\
            '_'+alg+'_'+scale_meth+'.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump([model, scaler], f)
        print('Trained model and scaler saved as '+model_file)
            
                        
    def exit_button():
        train_classifier_GUI.destroy()
        train_classifier_GUI.quit()
    
    
    
    tk.Button(train_classifier_GUI,
              text = "SELECT TRAINING SET",
              command = select_training_set_button) \
              .grid(row = 3,
                    column = 0,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')     
    
    tk.Button(train_classifier_GUI,
              text = "RUN",
              command = run_button) \
              .grid(row = 3,
                    column = 1,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
              
    tk.Button(train_classifier_GUI,
              text = "EXIT",
              command = exit_button) \
              .grid(row = 3,
                    column = 2,
                    padx=0,
                    pady=0,
                    sticky = 'W'+'E'+'N'+'S')
    
    train_classifier_GUI.title('Behavior Classifier Training GUI')
    
    curr_row = 0
    
    mainloop()


if __name__ == '__main__':
    
    try:
        train_behavior_classifier_GUI()
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)