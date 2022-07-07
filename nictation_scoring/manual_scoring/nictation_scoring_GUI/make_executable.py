# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:04:34 2022

Makes an executable version of the nictation scoring GUI.  It takes 5 min or 
so to run

WARNING: it will crash if the directory is open in Windows Explorer!

@author: PDMcClanahan
"""

# pip install pyinstaller
import PyInstaller




import PyInstaller.__main__

PyInstaller.__main__.run([
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\nictation_scoring_training\manual_scoring\nictation_scoring_GUI\nictation_scoring_GUI.py',
    '--onefile',
    '--windowed'
])


