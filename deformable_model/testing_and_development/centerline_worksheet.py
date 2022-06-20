# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:33:31 2021

@author: Temmerman Lab
"""

import cv2
import sys
import os
import copy
import time
import numpy as np
sys.path.append(os.path.split(__file__)[0])

import tracker_classes as tracker
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import matplotlib.pyplot as plt


# for find_centerline
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d

i = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw5.png'
bw = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
find_centerline(bw,'ridgeline',True)


bw_emb = np.zeros((np.shape(bw)[0]+100,np.shape(bw)[1]+100))
bw_emb[50:50+np.shape(bw)[0],50:50+np.shape(bw)[1]] = bw
bw_rot = rotate_image(bw_emb,-45)
find_centerline(bw_rot,'ridgeline',True)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def find_centerline(bw, method = 'ridgeline', debug = False):
        bw = np.uint8(bw)
        if method == 'ridgeline':
            N = 50
            # find the two ends

            # find the outline points
            outline = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(outline) > 1:
                outline = outline[0]
                if len(outline) > 1:
                    outline = outline[0]
            outline = np.squeeze(outline) # eliminate empty first and third dimensions
            
            xs = outline[:,0]
            ys = outline[:,1]
            
            # find the angles around the outline
            dxs = np.diff(xs,n=1,axis=0)
            dys = np.diff(ys,n=1,axis=0)
            
            angles = np.arctan2(dys,dxs) # range: (-pi,pi]
            dangles = np.diff(angles,n=1,axis = 0) # right turns are negative
            dangles = np.unwrap(dangles)
            
            # smooth angles and call this 'curvature'
            sigma = int(np.round(0.0125*len(xs)))
            curvature = gaussian_filter1d(dangles, sigma = sigma, mode = 'wrap')
            
            # the minimum curvature is likely to be either the head or tail
            end_1 = int(np.where(curvature == np.min(curvature))[0][0])
            
            # introduce a bias against finding the other end nearby
            ramp_up = np.linspace(0,1.5*curvature[end_1],int(0.9*(len(curvature)/2)))
            ramp_down = np.flipud(ramp_up)
            flat = np.zeros(int(np.shape(curvature)[0]-(np.shape(ramp_up)[0]+np.shape(ramp_down)[0])))
            ramp = np.concatenate((ramp_down,flat,ramp_up),axis = 0)
            bias = np.empty(len(curvature))
            if end_1 == 0:
                bias = ramp
            else:
                bias[0:end_1] = ramp[-end_1:]
                bias[end_1:] = ramp[0:len(ramp)-end_1]
            curvature_biased = curvature-bias
            end_2 = int(np.where(curvature_biased == np.min(curvature_biased))[0][0])
            
            
            # show the curvature
            if debug:
                from matplotlib.collections import LineCollection
                from matplotlib.colors import ListedColormap, BoundaryNorm
                color_weight = curvature
                points = np.array([xs,ys]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                fig,axs = plt.subplots(1,1,sharex=True,sharey=True)
                norm = plt.Normalize(color_weight.min(),color_weight.max())
                lc = LineCollection(segments, cmap = 'jet',norm = norm)
                lc.set_array(color_weight)
                lc.set_linewidth(3)
                line = axs.add_collection(lc)
                fig.colorbar(line, ax=axs)
                plt.title(r'Curvature (Interior Angle), $\sigma$ = '+str(sigma))
                plt.imshow(bw,cmap='gray')
                plt.plot(xs[end_1],ys[end_1],'ko',markerfacecolor = 'w')
                plt.plot(xs[end_2],ys[end_2],'ko',markerfacecolor = 'w')
                plt.show()
            
            
            # find the ridgeline points in middle of worm
            
            # find and smooth the distance transform
            
            dt = cv2.distanceTransform(bw,cv2.DIST_L2,maskSize=5)
            
            if debug:
                plt.imshow(dt,cmap = 'jet')
                plt.title('Distance Transform')
                plt.axis('off')
                plt.show()
            
            # find local maxima
            h_thr = 0
            inp = dt
            vert_pts = []
            for i in range(np.shape(inp)[1]):
                pts = find_peaks(inp[:,i],height = h_thr)
                for p in pts[0]:
                    vert_pts.append([p,i])
            
            horiz_pts = []
            for i in range(np.shape(inp)[0]):
                pts = find_peaks(inp[i,:],height = h_thr)
                for p in pts[0]:
                    horiz_pts.append([i,p])
            
            # combine the ends and ridge points in order
            ridge_pts = [pt for pt in horiz_pts if pt in vert_pts]
            cps = np.array([[ys[end_1],xs[end_1]]])
            while len(ridge_pts) != 0:
                dists = [np.linalg.norm(pt-cps[-1]) for pt in ridge_pts]
                cps = np.vstack((cps,ridge_pts.pop(int(np.where(dists == np.min(dists))[0][0]))))
            cps = np.vstack((cps,[ys[end_2],xs[end_2]]))
            
            if debug:
                plt.figure()
                plt.imshow(bw,cmap='gray')
                plt.title('Local Maxima in One and Both Directions')
                plt.plot(np.array(horiz_pts)[:,1],np.array(horiz_pts)[:,0],'.',color = [0,0,1])
                plt.plot(np.array(vert_pts)[:,1],np.array(vert_pts)[:,0],'.',color = [1,0,0])
                ridge_pts = [pt for pt in horiz_pts if pt in vert_pts]
                plt.plot(np.array(ridge_pts)[:,1],np.array(ridge_pts)[:,0],'.',color = [0,1,0])
                plt.axis('off')
                plt.show()
            
            
                plt.figure
                plt.imshow(bw,cmap='gray')
                for p in range(len(cps)):
                    plt.text(np.array(cps)[p,1],np.array(cps)[p,0],str(p),color = 'g',fontsize = 5)
                plt.title('Ordered Centerline Points')
                plt.axis('off')
                plt.show()
            
            # resample along the path of the centerline points
            
            # 1. find the length of the centerline, its current segments, and its segments
            # if it were resampled at regular intervals
            ds = [np.linalg.norm(cps[p+1]-cps[p]) for p in list(range(len(cps)-1))]
            cum_d = [np.sum(ds[0:p+1]) for p in range(len(ds))]
            cum_d.insert(0,0.0)
            steps = np.linspace(0,np.sum(ds),N)
            
            segs = [] # segment of the resampled point
            percs = [] # relative distance along the segment of the resampled point
            for s in steps[1:-1]:
                for d in range(len(cum_d)):
                    if s >= cum_d[d] and s < cum_d[d+1]:
                        segs.append(d)
                        percs.append((s-cum_d[d])/ds[d])
                        break   
            
            # 2. use this information to find the resampled points
            cpsr = []
            for s in range(len(segs)):
                start = cps[segs[s]]
                finish = cps[segs[s]+1]
                run = percs[s]*(finish - start)
                new_point = cps[segs[s]] + run
                cpsr.append(new_point)
            
            # 3. tack on the beginning and end points
            cpsr.insert(0,cps[0])
            cpsr.append(cps[-1])
            
            if debug:
                plt.figure()
                plt.imshow(bw,cmap='gray')
                plt.title('Resampled Centerline Points')
                plt.plot(np.array(cps)[:,1],np.array(cps)[:,0],'r.')
                plt.plot(np.array(cpsr)[:,1],np.array(cpsr)[:,0],'.',color = (0,1,0))
                plt.axis('off')
                plt.show()
            
            centerline = np.array(cpsr)
            centerline = np.fliplr(centerline)
        
        else:
            print('Error: centerline method not recognized')
        

# # try morphological thinning


# # functions
# def plt_img(im,label = ''):
#     if len(np.shape(im)) == 2:
#         plt.imshow(np.stack((im,im,im),2))
#     else:
#         plt.imshow(im)
#     ax = plt.gca();
#     # ax.set_xticks(np.arange(0, np.shape(im)[1], 1))
#     # ax.set_yticks(np.arange(0, np.shape(im)[0], 1))
#     # ax.set_xticklabels(np.arange(1,np.shape(im)[1]+1, 1))
#     # ax.set_yticklabels(np.arange(1, np.shape(im)[0]+1, 1))
#     # ax.set_xticks(np.arange(-.5, np.shape(im)[1], 1), minor=True)
#     # ax.set_yticks(np.arange(-.5, np.shape(im)[0], 1), minor=True)
#     # ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
#     if len(label) > 0:
#         #import pdb; pdb. set_trace()
#         t=plt.text(5,15,label,color='white',fontsize=12)

# img = copy.deepcopy(bw);

# strel1 = np.array([[-1, -1, -1],
#                    [0, 1, 0],
#                    [1, 1, 1]],dtype='int8')
# strel2 = np.array([[0, -1, -1],
#                    [1, 1, -1],
#                    [0, 1, 0]],dtype='int8')
# strels=(strel1,strel2,cv2.rotate(strel1,0),cv2.rotate(strel2,0),
#         cv2.rotate(strel1,1),cv2.rotate(strel2,1),
#         cv2.rotate(strel1,2),cv2.rotate(strel2,2))

# img_copy = copy.copy(img)
# label = 'morphological thinning'
# plt_img(img_copy,label)
# plt.pause(0.001); time.sleep(0.001)

# itr = 0
# while True:
#     ref_img = copy.copy(img_copy)
#     for strel in strels:
#         img_copy = cv2.bitwise_xor(img_copy,cv2.morphologyEx(img_copy, cv2.MORPH_HITMISS, strel))
#         #plt_img(img_copy,label)
#         #plt.pause(0.05); time.sleep(0.01)
#     #Show progress
#     #img_copy_3 = np.stack((img_copy,img_copy,img_copy),2)
#     plt_img(img_copy,label)
#     plt.pause(.001)
#     time.sleep(0.001)
#     # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
#     itr = itr+1
#     #print(itr)
#     if ~np.bitwise_xor(img_copy,ref_img).any():
#         break

# # show result
# img_copy_3 = np.stack((cv2.bitwise_or(img,img_copy),img,img),2)
# plt_img(img_copy_3,label)
# plt.pause(.05)
# time.sleep(0.01)