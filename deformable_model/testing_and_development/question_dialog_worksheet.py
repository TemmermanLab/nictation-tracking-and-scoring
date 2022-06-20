# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:46:11 2021

@author: Temmerman Lab
"""



import tkinter as tk
from tkinter import simpledialog

dialogue_box = tk.Tk()
dialogue_box.withdraw() # eliminates extra window

# returns None if you click cancel
answer = simpledialog.askfloat("Input", "Enter a number:",
                                 parent=dialogue_box,
                                 minvalue=0, maxvalue=100)

dialogue_box.destroy()
dialogue_box.quit()