# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:24 2022

This code is intended to give an idea of how cv2 plots lines onto a grid of
pixels for the purpose of correctly filling self-overlapping shapes drawn in
the Segmentation_GUI.

@author: Temmerman Lab
"""
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



# NB: the x and y returned by user mouse clicks are while number values

# img = np.zeros((10,10))

# clicks_x = [0,8]
# clicks_y = [8,2]

# l1=cv2.line(img, (clicks_x[0],clicks_y[0]), (clicks_x[1],clicks_y[1]), 255, 1)

# clicks_x = [7,2]
# clicks_y = [6,5]

# l2=cv2.line(img, (clicks_x[0],clicks_y[0]), (clicks_x[1],clicks_y[1]), 255, 1)



# plt.imshow(img,cmap = 'gray');
# plt.plot(clicks_x,clicks_y,'r-')
# plt.show()


# idea:
    
blank = np.zeros((100,100),np.uint8)

clicks_x = [31, 34, 38, 46, 55, 65, 78, 90, 92, 84, 74, 57, 39, 26, 23, 30, 39, 47, 61, 67, 74, 78, 80, 66, 61, 55, 50, 45, 42]
clicks_y = [82, 71, 58, 44, 33, 31, 31, 35, 54, 71, 79, 71, 65, 59, 50, 40, 46, 50, 56, 59, 58, 54, 42, 42, 48, 59, 75, 84, 86]


# image of lines connecting clicked points
lines_img = copy.copy(blank)
for i in range(len(clicks_x)):
    if i < len(clicks_x)-1:
        cv2.line(lines_img,(clicks_x[i],clicks_y[i]),(clicks_x[i+1],clicks_y[i+1]), 255, 1)
    else:
        cv2.line(lines_img,(clicks_x[i],clicks_y[i]),(clicks_x[0],clicks_y[0]), 255, 1)
   
plt.imshow(lines_img,cmap='gray'); plt.title('outline'); plt.show()


# image of holes
floodfill= copy.copy(lines_img)
h, w = floodfill.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(floodfill, mask, (0,0), 255);
floodfill_inv = cv2.bitwise_not(floodfill)
fill_poly = copy.copy(blank)
contours = np.flipud(np.rot90(np.vstack((clicks_x,clicks_y))))
fill_poly = cv2.fillPoly(fill_poly, pts =[contours], color=255)
holes_img = np.bitwise_xor(fill_poly,floodfill_inv)
holes_img = np.bitwise_xor(holes_img,lines_img)
plt.imshow(holes_img,cmap='gray'); plt.title('holes'); plt.show()


# image of right hand points
right_hand_points = copy.copy(blank)
for i in range(len(clicks_x)):
    # draw segment
    segment = copy.copy(blank)
    if i < len(clicks_x)-1:
        cv2.line(segment,(clicks_x[i],clicks_y[i]),(clicks_x[i+1],clicks_y[i+1]), 255, 1)
    else:
        cv2.line(segment,(clicks_x[i],clicks_y[i]),(clicks_x[0],clicks_y[0]), 255, 1)
    
    # eliminate points shared with other line segments
    other_lines = copy.copy(blank)
    for j in range(len(clicks_x)):
        if j != i:
            if j < len(clicks_x)-1:
                cv2.line(other_lines,(clicks_x[j],clicks_y[j]),(clicks_x[j+1],clicks_y[i+1]), 255, 1)
            else:
                cv2.line(other_lines,(clicks_x[j],clicks_y[j]),(clicks_x[0],clicks_y[0]), 255, 1)
    segment = segment - other_lines
    
    # find the normal direction to the right
    angle = np.arctan2((clicks_y[i+1]-clicks_y[i]),(clicks_x[i+1]-clicks_x[i])) # counterclockwise from origin to (1,0)
    norm_angle = angle - (np.pi/4)
    if norm_angle < 0:
        norm_angle = norm_angle + 2*np.pi
    
    # starting from one end, find the pixels lying to the right of the line segment, excluding those that make up part of the outline
    length = np.sqrt((clicks_y[i+1]-clicks_y[i])**2+(clicks_x[i+1]-clicks_x[i])**2)
    
    if length > 2.5:    
        # fit a parametric spline
        if i < len(clicks_x)-1:
            tck, u = interpolate.splprep([(clicks_x[i],clicks_x[i+1]), (clicks_y[i],clicks_y[i+1])], s=0)
        else:
            tck, u = interpolate.splprep([(clicks_x[i],clicks_x[0]), (clicks_y[i],clicks_y[0])], s=0)
            
        num_pts = int(length-2)
        pts = np.linspace(1,length-1,num_pts)
        pts_u = pts/length
        
        start_points = np.array(interpolate.splev(pts_new, tck))
        offset = np.array((np.sin(norm_angle),np.cos(norm_angle)))
        inside_points = start_points + offset
    
        # find points from which to 
        
        # use the spline to interpolate new extended centerline points
        if direction == 1:
            unew = np.linspace(0, 1+amount, self.N)
        elif direction == -1:
            unew = np.linspace(0-amount, 1, self.N)
        self.centerline = np.array(interpolate.splev(unew, tck))
    
        




