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
base_length = 253
half_width = 9
num_seg = 12
end1 = (.7,.9); end2 = (.9,.7,0)
middle = tuple(np.ones(num_seg-5))
width_factors = end1+middle+end2


# variable parameters
r = random.uniform(150,250)
c = random.uniform(150,250)
rot = random.uniform(0,360)
stretch = random.uniform(.25,1.05)
joint_angles = [0]
for a in range(num_seg-1):
    joint_angles.append(random.uniform(-45,45))
joint_angles.append(0)


# variable parameter limits
limits = (None,None,None,(.25,1.1),(-45,45),(-45,45),(-45,45),(-45,45),
          (-45,45),(-45,45),(-45,45),(-45,45),(-45,45),(-45,45),
          (-45,45))    


# for drawing
seg_length = (base_length*stretch) / num_seg
im2 = np.zeros((400,400),dtype = 'uint8')
color = 255
thickness = -1


# calculate worm points (x = col, y = row)
d_xs = np.zeros(num_seg+1)
v_xs = np.zeros(num_seg+1)
c_xs = np.zeros(num_seg+1)
d_ys = np.zeros(num_seg+1)
v_ys = np.zeros(num_seg+1)
c_ys = np.zeros(num_seg+1)


# calculate unrotated positions relative to the origin
for p in range(num_seg):
    curr_angle = np.sum(joint_angles[0:p+1])
    c_xs[p+1] = c_xs[p]+seg_length*-np.sin((-90+curr_angle)*(np.pi/180))
    c_ys[p+1] = c_ys[p]+seg_length*np.cos((-90+curr_angle)*(np.pi/180))
    d_xs[p+1] = c_xs[p+1]+width_factors[p]*half_width*-np.sin((curr_angle+0.5*joint_angles[p+1])*(np.pi/180))
    d_ys[p+1] = c_ys[p+1]+width_factors[p]*half_width*np.cos((curr_angle+0.5*joint_angles[p+1])*(np.pi/180))
    v_xs[p+1] = c_xs[p+1]+width_factors[p]*half_width*np.sin((curr_angle+0.5*joint_angles[p+1])*(np.pi/180))
    v_ys[p+1] = c_ys[p+1]+width_factors[p]*half_width*-np.cos((curr_angle+0.5*joint_angles[p+1])*(np.pi/180))
    
# translate the center point
c_off = c-c_xs[6]
r_off = r-c_ys[6]
d_xs = d_xs + c_off
v_xs = v_xs + c_off
c_xs = c_xs + c_off
d_ys = d_ys + r_off
v_ys = v_ys + r_off
c_ys = c_ys + r_off


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

worm_middle = int(num_seg/2)
rc = (copy.copy(c_xs[worm_middle]),copy.copy(c_ys[worm_middle]))
for p in range(len(d_xs)):
    c_xs[p],c_ys[p] = rotate(rc,(c_xs[p],c_ys[p]),rot)
    d_xs[p],d_ys[p] = rotate(rc,(d_xs[p],d_ys[p]),rot)
    v_xs[p],v_ys[p] = rotate(rc,(v_xs[p],v_ys[p]),rot)

# plot points
fig,ax = plt.subplots(figsize=(4,4))
ax.set_xlim(0,400)
ax.set_ylim(0,400)
plt.plot(c_xs,c_ys,'k.')
plt.plot(d_xs,d_ys,'b.')
plt.plot(v_xs,v_ys,'r.')
plt.show()

# create a bw image of points    
im2 = np.zeros((400,400),dtype = 'uint8')
color = 255
thickness = -1
for p in range(num_seg):
    xs = np.array([c_xs[p],d_xs[p],d_xs[p+1],c_xs[p+1],v_xs[p+1],v_xs[p]],dtype = 'int32')
    ys = np.array([c_ys[p],d_ys[p],d_ys[p+1],c_ys[p+1],v_ys[p+1],v_ys[p]],dtype = 'int32')
    contours = np.flipud(np.rot90(np.vstack((xs,ys))))
    cv2.fillPoly(im2, pts = [contours], color=255)

fig,ax = plt.subplots(figsize=(4,4))
ax.axis('off')
plt.imshow(im2,cmap = 'gray')
ax.invert_yaxis()
# ax.yaxis_inverted()
    
    
    
    
    
    
    
    
    
    
    


