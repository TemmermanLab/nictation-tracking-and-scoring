# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:11:40 2022

This script draws points to the left and right of a fictitious worm centerline
oriented at various angles.

@author: PDMcClanahan
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy
import sys
import os
import pickle

import time

def rotate(rot_center, p, angle):
    """Rotates a point counterclockwise around a point by an angle
    (degrees)"""
    angle = np.radians(angle)
    rcx, rcy = rot_center
    px, py = p

    x = rcx + np.cos(angle) * (px - rcx) - np.sin(angle) * (py - rcy)
    y = rcy + np.sin(angle) * (px - rcx) + np.cos(angle) * (py - rcy)
    return x, y



N = 5
centerline_main = np.array([[0,1,2,3,4,3],[0,1,0,1,0,-1]],dtype='float64')
width_factors = [0,.2,.2,.2,.2,0]

rot_ang = np.linspace(0,360,34)

for ra in rot_ang:

   
    centerline = copy.copy(centerline_main)
    for p in range(len(centerline[0])):
        xy = rotate((2,1),(centerline[0,p],centerline[1,p]),ra)
        centerline[0,p] = xy[0]
        centerline[1,p] = xy[1]
    
    # calculate joint angles
    dy = np.diff(centerline[1])
    dx = np.diff(centerline[0])
    segment_angles = np.arctan2(dy,dx)
    segment_angles = np.unwrap(segment_angles)
    joint_angles = np.convolve(segment_angles, np.ones(2), 'valid') / 2
    # print(joint_angles[1]-np.radians(ra))
    print(ra)
    
    #joint_angles = (joint_angles + np.pi) % (2 * np.pi)
    #joint_angles = np.unwrap(joint_angles)
    
    RHS_x = centerline[0][1:-1] + width_factors[1:-1] * np.sin(joint_angles)
    RHS_y = centerline[1][1:-1] - width_factors[1:-1] * np.cos(joint_angles)
    
    LHS_x = centerline[0][1:-1] - width_factors[1:-1] * np.sin(joint_angles)
    LHS_y = centerline[1][1:-1] + width_factors[1:-1] * np.cos(joint_angles)
    
    

    
    
    
   
    # plot
    fig,axes = plt.subplots()
    axes.plot(centerline[0][0],centerline[1][0],'k.')
    axes.plot(centerline[0],centerline[1],'k-')
    axes.plot(RHS_x,RHS_y,'b-')
    axes.plot(LHS_x,LHS_y,'r-')
    axes.set_aspect('equal')
    fig.show()





