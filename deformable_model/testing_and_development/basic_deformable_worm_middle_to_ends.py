# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:22:07 2021

@author: Temmerman Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy


# constant parameters
total_length = 253
half_width = 9
seg_per_side = 6
end_taper = (0.9,0.6,0)
width_factors = np.ones(seg_per_side)
width_factors[-len(end_taper):] = end_taper


# variable parameters
r = random.uniform(150,250)
c = random.uniform(150,250)
rot = random.uniform(0,360)
stretch = random.uniform(.5,1.1)
joint_angles_left = []
joint_angles_left.append(random.uniform(-22.5,22.5))
for a in range(1,seg_per_side):
    joint_angles_left.append(random.uniform(-45,45))
joint_angles_right = []
joint_angles_right.append(random.uniform(-22.5,22.5))
for a in range(1,seg_per_side):
    joint_angles_right.append(random.uniform(-45,45))


# variable parameter limits
limits = [None,None,None,(.25,1.1)]
for a in range(2*seg_per_side):
    limits = limits+[(-45,45)] 


# for drawing
seg_length = total_length / (2*seg_per_side)
im2 = np.zeros((400,400),dtype = 'uint8')
color = 255
thickness = -1


# calculate worm points
# rows: centerline xs, ys; dorsal xs, ys; ventral xs, ys
# cols: segments
pts = np.zeros((6,seg_per_side*2+1))



# calculate RHS points from origin
offset = seg_per_side+1
for p in range(seg_per_side):
    curr_angle = np.sum(joint_angles_right[0:p+1])
    print(curr_angle)
    if p < seg_per_side-1:
        alpha = curr_angle+0.5*joint_angles_right[p+1] # tangent at segment point used to draw sides
    else:
        alpha = curr_angle
    # centerline points
    pts[0,offset+p] = pts[0,offset+p-1]+seg_length*stretch*np.cos(curr_angle*(np.pi/180))
    pts[1,offset+p] = pts[1,offset+p-1]+seg_length*stretch*np.sin(curr_angle*(np.pi/180))
    # "dorsal" points
    pts[2,offset+p] = pts[0,offset+p]+width_factors[p]*half_width*-np.sin((alpha)*(np.pi/180))
    pts[3,offset+p] = pts[1,offset+p]+width_factors[p]*half_width*np.cos((alpha)*(np.pi/180))
    # "ventral" points
    pts[4,offset+p] = pts[0,offset+p]-width_factors[p]*half_width*-np.sin((alpha)*(np.pi/180))
    pts[5,offset+p] = pts[1,offset+p]-width_factors[p]*half_width*np.cos((alpha)*(np.pi/180))

# calculate LHS points from origin
for p in range(seg_per_side):
    curr_angle = np.sum(joint_angles_left[0:p+1])
    print(curr_angle)
    if p < seg_per_side-1:
        alpha = curr_angle+0.5*joint_angles_left[p+1] # tangent at segment point used to draw sides
    else:
        alpha = curr_angle
    # centerline points
    pts[0,seg_per_side-1-p] = pts[0,seg_per_side-p]-seg_length*stretch*np.cos(curr_angle*(np.pi/180))
    pts[1,seg_per_side-1-p] = pts[1,seg_per_side-p]-seg_length*stretch*np.sin(curr_angle*(np.pi/180))
    # "dorsal" points
    pts[4,seg_per_side-1-p] = pts[0,seg_per_side-1-p]-width_factors[p]*half_width*-np.sin((alpha)*(np.pi/180))
    pts[5,seg_per_side-1-p] = pts[1,seg_per_side-1-p]-width_factors[p]*half_width*np.cos((alpha)*(np.pi/180))
    # "ventral" points
    pts[2,seg_per_side-1-p] = pts[0,seg_per_side-1-p]+width_factors[p]*half_width*-np.sin((alpha)*(np.pi/180))
    pts[3,seg_per_side-1-p] = pts[1,seg_per_side-1-p]+width_factors[p]*half_width*np.cos((alpha)*(np.pi/180))

# calculate the middle points
alpha = np.mean((joint_angles_left[0],joint_angles_right[0]))
pts[4][seg_per_side] = -width_factors[0]*half_width*-np.sin((alpha)*(np.pi/180))
pts[5][seg_per_side] = -width_factors[0]*half_width*np.cos((alpha)*(np.pi/180))
pts[2][seg_per_side] = width_factors[0]*half_width*-np.sin((alpha)*(np.pi/180))
pts[3][seg_per_side] = width_factors[0]*half_width*np.cos((alpha)*(np.pi/180))


   
# translate the center point
c_off = c-pts[0][seg_per_side]
r_off = r-pts[1][seg_per_side]
for r in range(0,6,2):
    pts[r] = pts[r] + c_off
    pts[r+1] = pts[r+1] + r_off



# rotate about the center point
def rotate(rot_center, p, angle):
    """
    Rotate a point counterclockwise around a point by an angle (degrees).
    """
    angle = np.radians(angle)
    rcx, rcy = rot_center
    px, py = p

    x = rcx + np.cos(angle) * (px - rcx) - np.sin(angle) * (py - rcy)
    y = rcy + np.sin(angle) * (px - rcx) + np.cos(angle) * (py - rcy)
    return x, y


rc = (copy.copy(pts[0][seg_per_side]),copy.copy(pts[1][seg_per_side]))
for r in range(0,6,2):
    for p in range(np.shape(pts)[1]):
        pts[r][p],pts[r+1][p] = rotate(rc,(pts[r][p],pts[r+1][p]),rot)
        

# plot points
fig,ax = plt.subplots(figsize=(4,4))
# ax.set_xlim(0,400)
# ax.set_ylim(0,400)
start = 0; stop = 13
plt.plot(pts[0][start:stop],pts[1][start:stop],'k.')
plt.plot(pts[2][start:stop],pts[3][start:stop],'b.')
plt.plot(pts[4][start:stop],pts[5][start:stop],'r.')
ax.axis('equal')
plt.show()

# create a bw image of points    
im2 = np.zeros((400,400),dtype = 'uint8')
color = 255
thickness = -1
for p in range(np.shape(pts)[1]-1):
    xs = np.array([pts[0][p],pts[2][p],pts[2][p+1],pts[0][p+1],pts[4][p+1],pts[4][p]],dtype = 'int32')
    ys = np.array([pts[1][p],pts[3][p],pts[3][p+1],pts[1][p+1],pts[5][p+1],pts[5][p]],dtype = 'int32')
    contours = np.flipud(np.rot90(np.vstack((xs,ys))))
    cv2.fillPoly(im2, pts = [contours], color=255)

fig,ax = plt.subplots(figsize=(4,4))
ax.axis('off')
plt.imshow(im2,cmap = 'gray')
ax.invert_yaxis()
# ax.yaxis_inverted()
    
    
    
    
    
    
    
    
    
    
    


