# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:52:38 2022

@author: Temmerman Lab
"""
import os
import sys
sys.path.append(os.path.split(__file__)[0])
import tracker_classes as tracker


t = tracker.Tracker(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_cropped\super_crop.avi')
t.track()