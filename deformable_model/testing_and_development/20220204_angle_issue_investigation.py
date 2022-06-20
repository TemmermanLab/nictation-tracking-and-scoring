# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:42:08 2022

The end_1_angle of worm 36 in the video
Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi was greater than
zero, which should be impossible and could indicate a bug in the code.

The problem was a small spur in the bw segmentation causing the outline to
double back on itself. I introduced code to remove such points from the
outline

@author: Temmerman Lab
"""

# modules
import cv2
import sys
import os
import numpy as np
import copy

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d

sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation')
import mrcnn_module as mrcnn
import tracker_classes as Tracker

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# parameters
model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Steinernema\20220127_full_frame_Sc_on_udirt_4.pt'
model, device = mrcnn.prepare_model(model_file) 
scale_factor = 0.5
bw_thr = 100
k_sig = 1.5
k_size = (round(k_sig*3)*2+1,round(k_sig*3)*2+1)
edge_proximity_cutoff = 10
centerline_method = 'ridgeline'
area_bnds = (1200,6500)

# functions
def find_centerline(bw, method = 'ridgeline', debug = False):
    '''Takes a binary image of a worm and returns the centerline. The ends
    are detected by finding two distant minima of the smoothed interior 
    angle. Centerline points are detected by finding points that are local
    minima of the distance transform in both the x and y direction. These
    points are resampled and returned along with the smoothed interior 
    angles at each end.'''
    bw = np.uint8(bw)
    if method == 'ridgeline':
        
        #bw[2,0] = 0
        N = 50
        # find the two ends

        # find the outline points
        outline = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(outline) > 1:
            outline = outline[0]
            if len(outline) > 1:
                outline = outline[0]
        outline = np.squeeze(outline) # eliminate empty first and third dimensions
        
        # find and eliminate one-connected points
        # NB: this solves a bug that occurs when the outline doubles back on
        # itself, messing up the unwrapping of the outline angles. It
        # eliminates one-connected points (spurs) in the outline until none remain (in
        # case there is a spur more than one point long). Alternatives of
        # morphological opening and closing also work. Another approach would be
        # to use a 3x3 kernel to look for and eliminate 1-connected pixels.
        # However, these approaches could all fail if the bw object contains
        # an isthmus.
        search = True
        while search:
            found = False
            for p in reversed(range(len(outline))):
                pt_before = outline[p-2]
                pt_after = outline[p]
                if sum(pt_before == pt_after) == 2:
                    outline = np.delete(outline,[p-1,p],axis = 0)
                    found = True
            if found == 0:
                search = False
                
        xs = outline[:,0]
        ys = outline[:,1]
        
        # find the angles around the outline
        dxs = np.diff(xs,n=1,axis=0)
        dys = np.diff(ys,n=1,axis=0)
        
        angles = np.arctan2(dys,dxs) # range: (-pi,pi]
        dangles = np.diff(angles,n=1,axis = 0) # right turns are negative
        dangles = np.unwrap(dangles)
        # dangles[np.where(dangles > 2*np.pi)[0]] = dangles[np.where(dangles > 2*np.pi)[0]] - 2*np.pi
        
        # smooth angles and call this 'curvature'
        sigma = int(np.round(0.0125*len(xs)))
        curvature = gaussian_filter1d(dangles, sigma = sigma, mode = 'wrap')
        
        # the minimum curvature is likely to be either the head or tail
        end_1 = int(np.where(curvature == np.min(curvature))[0][0])
        curvature_end_1 = curvature[end_1]
        
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
        curvature_end_2 = curvature[end_2]
        
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
            # plt.xlim((0,50))
            # plt.ylim((0,10))
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
    
    return centerline, curvature_end_1, curvature_end_2

def flag_bad_centerline(cline, scale = 1, max_length = 1000):
    '''Flags a centerline if it is too long or kinky'''
    flag = 0
    cline = cline.astype('int16')
    # determine if the centerline is too long
    length = 0
    for s in range(np.shape(cline)[0]-1):
        length += np.linalg.norm(cline[s+1]-cline[s])
    #print('length: '+str(length))
    if length > 350:
        flag = 1
    
    # determine if the centerline has kinks
    dxs = np.diff(cline[:,0],n=1,axis=0)
    dys = np.diff(cline[:,1],n=1,axis=0)
    angles = np.arctan2(dys,dxs)
    angles = np.diff(angles)
    angles = np.unwrap(angles)
    angles = np.degrees(angles)
    #print('max_angle: '+str(np.max(np.abs(angles))))
    if np.max(np.abs(angles)) > 91:
        flag = 1
    
    # determine if the centerline crosses itself more than once
    # 1. Determine the point of intersection (if not parallel)
    # 2. Determine if that point is within both of the potentially-
    # crossing line segments.
    
    return flag





# load frame
for f in range(100):
    print(f)
    vid = cv2.VideoCapture(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi')
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # segment frame
    diff = mrcnn.segment_full_frame(img, model, device, scale_factor)
    smooth = cv2.GaussianBlur(diff,k_size,k_sig,cv2.BORDER_REPLICATE)
    thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
    
    cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    cc_map = np.uint8(cc[1]); 
    cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)
    
    # eliminate objects that are too big, too small, or touch the boundary
    centroids = list()
    centerlines = list()
    centerline_flags = list()
    angles_end_1 = list()
    angles_end_2 = list()
    for cc_i in cc_is:
        cc_sz = cc[2][cc_i][4]
        if cc_sz > area_bnds[0] and cc_sz < area_bnds[1]:
            hits_edge = False
            obj_inds_r = np.where(cc[1]==cc_i)[0]
            obj_inds_c = np.where(cc[1]==cc_i)[1]
    
            if np.min(obj_inds_r) <= edge_proximity_cutoff or np.min(obj_inds_c) <= edge_proximity_cutoff:
                hits_edge = True
            elif np.max(obj_inds_r) >= np.shape(cc[1])[0]-1-edge_proximity_cutoff or np.max(obj_inds_c) >= np.shape(cc[1])[1]-1-edge_proximity_cutoff:
                hits_edge = True
                
            if hits_edge is False:
                centroids.append(copy.deepcopy(cc[3][cc_i]))
                
                # find the centerline
                if centerline_method != 'none':
                    bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                    bw_w[np.where(bw_w == cc_i)]=255
                    bw_w[np.where(bw_w!=255)]=0
                    centerline, angle_end_1, angle_end_2 = find_centerline(bw_w,centerline_method)
                    centerline = np.uint16(np.round(centerline))
                    centerline_flag = flag_bad_centerline(centerline)
                    centerline[:,0] += cc[2][cc_i][0]
                    centerline[:,1] += cc[2][cc_i][1]
                    centerline = centerline[np.newaxis,...]
                    centerlines.append(copy.copy(centerline))
                    centerline_flags.append(centerline_flag)
                    angles_end_1.append(angle_end_1)
                    angles_end_2.append(angle_end_2)
                    if angle_end_1 > 0:
                        print(f)
                        print(cc_i)
                        break
# run centerline code

# the problem is in frame 94, object 15 (9 after QC)















 