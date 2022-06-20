# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:54:03 2021

@author: Temmerman Lab
"""

import cv2
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def calc_loss1(X):
    im2 = make_image(X)
    loss = calc_loss2(im1,im2)
    return loss


def calc_loss2(im1,im2):
    im1 = np.int16(im1); im2 = np.int16(im2)
    im_comb = im1+im2
    intersection = np.sum(np.where(im_comb == 510)[0])
    union = np.sum(np.where(im_comb == 255)[0]) + intersection
    IoU = intersection / union
    loss = 1- IoU
    return loss


def make_image(X):
    im = np.zeros((100,100),dtype = 'uint8')
    center_coordinates = (int(X[0]),int(X[1]))
    axes_length = (int(X[2]),int(X[3]))
    angle = X[4]
    start_angle = 0
    end_angle = 360
    color = 255
    thickness = -1
    im = cv2.ellipse(im, center_coordinates, axes_length,
           angle, start_angle, end_angle, color, thickness)
    return im

def show_IoU(im1,im2):
    loss = calc_loss2(im1,im2); IoU = 1-loss
    im1 = np.int16(im1); im2 = np.int16(im2)
    im_comb = (im1+im2)/2
    fig,ax = plt.subplots()
    
    plt.imshow(im_comb,cmap = 'gray')
    ax.axis('off')
    plt.title('Overlap')
    plt.text(5,5,'IoU = '+str(round(IoU,2)),color = 'white')
    
    # image for demo video    
    canvas = FigureCanvas(fig) # for demo vid
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    image = image[:,:,0]; image = np.squeeze(image)
    
    plt.show()
    return IoU, image
    
def calc_grad(X,loss_fun = calc_loss1,steps = 1):
    if len(np.shape(steps))==0:
        steps = np.ones(len(X))
    dldX = []
    for i in range(len(X)):
        X[i] = X[i]-steps[i]
        loss1 = loss_fun(X)
        X[i] = X[i]+(2*steps[i])
        loss2 = loss_fun(X)
        X[i] = X[i]-steps[i]
        dldX.append(loss2-loss1)
    return dldX

# from https://realpython.com/gradient-descent-algorithm-python/
def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06, demo_vid = True):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(vector))
        # if np.all(np.abs(diff) <= tolerance):
        #     break
        vector += diff
        im_moving = make_image(vector)
        IoU, image = show_IoU(im1,im_moving)
        if IoU > 0.97:
            break
        if demo_vid == True:
            if _ == 0:
                stack = image
            elif _ == 1:
                stack = np.stack((stack,image),-1)
            else:
                stack = np.concatenate((stack,np.expand_dims(image,-1)),-1)
            
    return vector, stack

def write_demo_vid(stack, save_dir):
    vid_file = datetime.now().strftime("%Y%m%d%H%M%S") +'_grad_descent_reg_demo.avi'
    out_w = np.shape(stack)[1]
    out_h = np.shape(stack)[0]
    v_out = cv2.VideoWriter(save_dir+'\\'+vid_file,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10, (out_w,out_h), 0)
    
    for f in range(np.shape(stack)[2]):
        v_out.write(np.squeeze(stack[:,:,f]))
        
    v_out.release()

try:
    im1 = np.zeros((100,100),dtype = 'uint8')
    cv2.circle(im1,(50,50),25,color = 255,thickness = -1)
    
    im2 = np.zeros((100,100),dtype = 'uint8')
    
    e2row_0 = random.uniform(40,60)
    e2col_0 = random.uniform(40,60)
    e2x_0 = random.uniform(20,30)
    e2y_0 = random.uniform(20,30)
    e2rot_0 = random.uniform(0,360)
    
    center_coordinates = (int(e2row_0),int(e2col_0))
    axes_length = (int(e2x_0),int(e2y_0))
    angle = e2rot_0
    start_angle = 0
    end_angle = 360
    color = 255
    thickness = -1
    
    X0 = [e2row_0,e2col_0,e2x_0,e2y_0,e2rot_0]
    
    # cv2.ellipse(im2,center_coordinates=(e2row_0,e2col_0),axesLength = (e2x_0,e2y_0),angle = e2rot_0, color = 255, thickness = -1)
    im2 = cv2.ellipse(im2, center_coordinates, axes_length,
               angle, start_angle, end_angle, color, thickness)
    
    
    show_IoU(im1,im2)
    XF, demo_stack = gradient_descent(calc_grad,X0,10,200,demo_vid = True)
    save_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files'
    write_demo_vid(demo_stack,save_dir)

except:
    import pdb
    import sys
    import traceback
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)


# loss_0 = calc_loss(im1,im2)
# Reading an image in default mode

   
# Using cv2.ellipse() method
# Draw a ellipse with red line borders of thickness of 5 px
