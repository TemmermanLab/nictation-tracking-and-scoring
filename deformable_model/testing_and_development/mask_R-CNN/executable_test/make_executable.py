# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:32:02 2022

@author: Temmerman Lab
"""

import PyInstaller




import PyInstaller.__main__

PyInstaller.__main__.run([
    r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test\segment_one_frame.py',
    '--onefile',
    '--windowed'
])