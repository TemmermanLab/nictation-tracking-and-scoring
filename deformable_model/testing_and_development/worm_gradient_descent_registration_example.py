# -*- coding: utf-8 -*-
"""

Created on Thu Dec  2 14:54:03 2021
@author: P. D. McClanahan (pdmcclanahan@gmail.com)
"""

import cv2
import random
import copy
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def calc_loss1(X, seg_length, half_width, metric = 'IoU'):
    im2 = make_image(X, seg_length, half_width)
    if metric == 'IoU':    
        loss = calc_loss2_IoU(im1,im2)
    elif metric == 'target_overlap':
        loss = calc_loss2_TO(im1,im2)
        
    return loss


def calc_loss2_IoU(im1,im2):
    im1 = np.int16(im1); im2 = np.int16(im2)
    im_comb = im1+im2
    intersection = np.sum(np.where(im_comb == 510)[0])
    union = np.sum(np.where(im_comb == 255)[0]) + intersection
    IoU = intersection / union
    loss = 1- IoU
    return loss

def calc_loss2_TO(im1,im2):
    # im1 is the target
    im1 = np.int16(im1); im2 = np.int16(im2)
    im_comb = im1+im2
    intersection = np.sum(np.where(im_comb == 510)[0])
    target_area = np.sum(np.where(im1 == 255)[0]) + intersection
    TO = intersection / target_area
    loss = 1 - TO
    return loss

def make_image(X, seg_length, half_width, plot = False):
    im = np.zeros((400,400),dtype = 'uint8')
    r = X[0]; c = X[1]; rot = X[2]; stretch = X[3]
    seg_per_side = int((len(X)-4)/2)
    joint_angles_left = X[4:4+seg_per_side]
    joint_angles_right = X[4+seg_per_side:4+2*seg_per_side]
    
    # calculate worm points
    # rows: centerline xs, ys; dorsal xs, ys; ventral xs, ys
    # cols: segments
    pts = np.zeros((6,seg_per_side*2+1))
    
    # calculate RHS points from origin
    offset = seg_per_side+1
    for p in range(seg_per_side):
        curr_angle = np.sum(joint_angles_right[0:p+1])
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
    
    
    rc = (copy.copy(pts[0][seg_per_side]),copy.copy(pts[1][seg_per_side]))
    for r in range(0,6,2):
        for p in range(np.shape(pts)[1]):
            pts[r][p],pts[r+1][p] = rotate(rc,(pts[r][p],pts[r+1][p]),rot)
            
    
    # # plot points
    # fig,ax = plt.subplots(figsize=(4,4))
    # # ax.set_xlim(0,400)
    # # ax.set_ylim(0,400)
    # start = 0; stop = 13
    # plt.plot(pts[0][start:stop],pts[1][start:stop],'k.')
    # plt.plot(pts[2][start:stop],pts[3][start:stop],'b.')
    # plt.plot(pts[4][start:stop],pts[5][start:stop],'r.')
    # ax.axis('equal')
    # plt.show()
    
    # create a bw image of points    
    im = np.zeros((400,400),dtype = 'uint8')
    color = 255
    thickness = -1
    for p in range(np.shape(pts)[1]-1):
        xs = np.array([pts[0][p],pts[2][p],pts[2][p+1],pts[0][p+1],pts[4][p+1],pts[4][p]],dtype = 'int32')
        ys = np.array([pts[1][p],pts[3][p],pts[3][p+1],pts[1][p+1],pts[5][p+1],pts[5][p]],dtype = 'int32')
        contours = np.flipud(np.rot90(np.vstack((xs,ys))))
        cv2.fillPoly(im, pts = [contours], color=255)
    
    return im


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


def show_IoU(im1,im2,iteration = -1):
    loss = calc_loss2_IoU(im1,im2); IoU = 1-loss
    im1 = np.int16(im1); im2 = np.int16(im2)
    im_comb = (im1+im2)/2
    fig,ax = plt.subplots()
    
    plt.imshow(im_comb,cmap = 'gray')
    ax.axis('off')
    plt.title('Overlap')
    plt.text(5,20,'IoU = '+str(round(IoU,2)),color = 'white')
    if iteration > -1:
        plt.text(5,40,'Iter = '+str(round(iteration,2)),color = 'white')
    # image for demo video    
    canvas = FigureCanvas(fig) # for demo vid
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    image = image[:,:,0]; image = np.squeeze(image)
    
    plt.show()
    return IoU, image
    
def calc_grad(X,  seg_length, half_width, loss_fun = calc_loss1, metric = 'IoU', steps = 1):
    # if len(np.shape(steps))==0:
    #     steps = np.ones(len(X))
    #import pdb; pdb.set_trace()
    steps1 = [1,1,1,.1]
    steps2 = []
    for s in range(seg_per_side*2):steps2.append(5)
    steps = np.array(steps1+steps2)
    del steps1, steps2, s
    
    dldX = []
    for i in range(len(X)):
        X[i] = X[i]-steps[i]
        loss1 = loss_fun(X, seg_length, half_width, metric)
        X[i] = X[i]+(2*steps[i])
        loss2 = loss_fun(X, seg_length, half_width, metric)
        X[i] = X[i]-steps[i]
        dldX.append(loss2-loss1)
    return dldX

# from https://realpython.com/gradient-descent-algorithm-python/
def gradient_descent(gradient_fun, start, limits, learn_rate, n_iter=50, num_seg = 12, seg_length = 10.5, half_width = 9, demo_vid = True):
    vector = start
    slow_downs = 1
    metric = 'target_overlap'
    for _ in range(n_iter):
        #import pdb; pdb.set_trace()
        diff = -learn_rate * np.array(gradient_fun(vector, seg_length, half_width, calc_loss1, metric))
        vector += diff
        #print(vector[])
        for x in range(len(vector)):
            if limits[x] is not None:
                if vector[x] < limits[x][0]:
                    vector[x] = limits[x][0]
                elif vector[x] > limits[x][1]:
                    vector[x] = limits[x][1]
        im_moving = make_image(vector, seg_length, half_width, False)
        IoU, image = show_IoU(im1,im_moving,_)
        print(_)
        if IoU > 0.5 and slow_downs > 0:
            learn_rate = learn_rate/10
            slow_downs -= 1
        if IoU > 0.95:
            break
        if demo_vid == True:
            if _ == 0:
                stack = image
            elif _ == 1:
                stack = np.stack((stack,image),-1)
            else:
                stack = np.concatenate((stack,np.expand_dims(image,-1)),-1)
    
    if demo_vid == False:
        stack = []
    
    return vector, stack
    

def write_demo_vid(stack, save_dir):
    vid_file = datetime.now().strftime("%Y%m%d%H%M%S") +'_grad_descent_worm_demo.avi'
    
    out_w = np.shape(stack)[1]
    out_h = np.shape(stack)[0]
    v_out = cv2.VideoWriter(save_dir+'\\'+vid_file,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10, (out_w,out_h), 0)
    
    for f in range(np.shape(stack)[2]):
        v_out.write(np.squeeze(stack[:,:,f]))
        
    v_out.release()


try:
    im1 = np.zeros((400,400),dtype = 'uint8')
    # worm_img = cv2.imread(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw1.png',cv2.IMREAD_GRAYSCALE)
    worm_img = cv2.imread(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw5.png',cv2.IMREAD_GRAYSCALE)
    shift_x = round((np.shape(im1)[1]/2)-(np.shape(worm_img)[1]/2))
    shift_y = round((np.shape(im1)[0]/2)-(np.shape(worm_img)[0]/2))
    
    im1[shift_y:shift_y+np.shape(worm_img)[0],shift_x:shift_x+np.shape(worm_img)[1]] = worm_img
    
   
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
    for row in range(0,6,2):
        pts[row] = pts[row] + c_off
        pts[row+1] = pts[row+1] + r_off
    
    
    rc = (copy.copy(pts[0][seg_per_side]),copy.copy(pts[1][seg_per_side]))
    for row in range(0,6,2):
        for p in range(np.shape(pts)[1]):
            pts[row][p],pts[row+1][p] = rotate(rc,(pts[row][p],pts[row+1][p]),rot)
            
    
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
    
    
    # prepare for registration
    X0 = [r,c,rot,stretch]+joint_angles_left + joint_angles_right
    lr1 = [1,1,1,.01]; lr2 = []
    for s in range(seg_per_side*2): lr2.append(0.1)
    lr = np.array(lr1+lr2)*100
    del lr1, lr2, s
    
    # registration
    num_seg = seg_per_side*2
    show_IoU(im1,im2)
    demo_vid = False
    XF, demo_stack = gradient_descent(calc_grad,X0,limits,lr,300,num_seg,seg_length,half_width,demo_vid)
    save_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files'
    if demo_vid: write_demo_vid(demo_stack,save_dir)

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
