# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:21:40 2022

This module contains methods for finding the centerline of a 
worm based on its binary segmentation.  The method employed here involves
locating the ridgeline of the the distance transform of the worm

@author: Temmerman Lab
"""
import numpy as np
import cv2
import copy
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_edt as dist_trans
from scipy import interpolate
from scipy.spatial import distance

    
def find_centerline(bw, method = 'ridgeline', debug = False):
    '''Takes a binary image of a worm and returns the centerline. The ends
    are detected by finding two distant minima of the smoothed interior 
    angle. Centerline points are detected by finding points that are local
    minima of the distance transform in both the x and y direction. These
    points are resampled and returned along with the smoothed interior 
    angles at each end.'''
    
    #bw = np.uint8(bw)
    
    # get rid of small holes
    strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    bw = cv2.dilate(bw.astype('uint8'), strel)
    bw = cv2.erode(bw, strel, borderType = cv2.BORDER_CONSTANT, 
                   borderValue = 0)

    if method == 'ridgeline':
        N = 50
        # find the two ends

        # find the outline points
        outline = cv2.findContours(bw, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
        if len(outline) > 1:
            outline = outline[0]
            if len(outline) > 1:
                outline = outline[0]
        # eliminate empty first and third dimensions
        outline = np.squeeze(outline) 
        # needed to have a curvature for every point
        outline = np.vstack((outline[-1:,:],outline,outline[0:1,:])) 
        
        # find and eliminate one-connected points
        '''NB: this solves a bug that occurs when the outline doubles back
        on itself, messing up the unwrapping of the outline angles. It
        eliminates one-connected points (spurs) in the outline until none
        remain (in case there is a spur more than one point long). 
        Alternatives of morphological opening and closing also work. 
        Another approach would be to use a 3x3 kernel to look for and 
        eliminate 1-connected pixels. However, these approaches could all
        fail if the bw object contains an isthmus.'''
        search = True
        while search:
            found = False
            for p in reversed(range(2,len(outline))):
                # avoids rare index error if points deleted
                if p < len(outline): 
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
        #dangles_unwr = np.unwrap(dangles)
        dangles_unwr = unwrap(dangles)
        
        # smooth angles and call this 'curvature'
        sigma = int(np.round(0.0125*len(xs)))
        if sigma == 0:
            sigma = 1
        curvature = gaussian_filter1d(dangles_unwr, sigma = sigma, 
                                      mode = 'wrap')
        
        # the minimum curvature is likely to be either the head or tail
        end_1 = int(np.where(curvature == np.min(curvature))[0][0])
        curvature_end_1 = curvature[end_1]
        if curvature_end_1 < -6 or curvature_end_1 > 0:
            import pdb; pdb.set_trace()
        
        # introduce a bias against finding the other end nearby
        ramp_up = np.linspace(0,1.5*curvature[end_1],
                              int(0.9*(len(curvature)/2)))
        ramp_down = np.flipud(ramp_up)
        flat = np.zeros(int(np.shape(curvature)[0]- \
                        (np.shape(ramp_up)[0]+np.shape(ramp_down)[0])))
        ramp = np.concatenate((ramp_down,flat,ramp_up),axis = 0)
        bias = np.empty(len(curvature))
        if end_1 == 0:
            bias = ramp
        else:
            bias[0:end_1] = ramp[-end_1:]
            bias[end_1:] = ramp[0:len(ramp)-end_1]
        curvature_biased = curvature-bias
        end_2 = int(np.where(curvature_biased == np.min(
            curvature_biased))[0][0])
        curvature_end_2 = curvature[end_2]
        xs = xs[1:-1]; ys = ys[1:-1] # curvature not defined at endpoints
        
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
            plt.title(r'Curvature (Interior Angle), $\sigma$ = '+ \
                      str(sigma))
            plt.imshow(bw,cmap='gray')
            plt.plot(xs[end_1],ys[end_1],'ko',markerfacecolor = 'w')
            plt.plot(xs[end_2],ys[end_2],'ko',markerfacecolor = 'w')
            # plt.xlim(-1,10)
            # plt.ylim(10,-1)
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
            # # fails, cps[0] is always equal to cps[-1], idk why                
            # dists = [np.linalg.norm(pt-cps[-1]) for pt in ridge_pts]

            # works
            dists = []
            for pt in ridge_pts:
                dists.append(np.linalg.norm(pt-cps[-1]))
            
            cps = np.vstack((cps,ridge_pts.pop(int(np.where(
                dists == np.min(dists))[0][0]))))

        cps = np.vstack((cps,[ys[end_2],xs[end_2]]))
        
        if debug:
            plt.figure()
            plt.imshow(bw,cmap='gray')
            plt.title('Local Maxima in One and Both Directions')
            plt.plot(np.array(horiz_pts)[:,1],np.array(horiz_pts)[:,0],
                     '.',color = [0,0,1])
            plt.plot(np.array(vert_pts)[:,1],np.array(vert_pts)[:,0],
                     '.',color = [1,0,0])
            ridge_pts = [pt for pt in horiz_pts if pt in vert_pts]
            plt.plot(np.array(ridge_pts)[:,1],np.array(ridge_pts)[:,0],
                     '.',color = [0,1,0])
            plt.axis('off')
            plt.show()
        
        
            plt.figure
            plt.imshow(bw,cmap='gray')
            for p in range(len(cps)):
                plt.text(np.array(cps)[p,1],np.array(cps)[p,0],str(p),
                         color = 'g',fontsize = 7)
            plt.title('Ordered Centerline Points')
            plt.axis('off')
            plt.show()
        
        # resample along the path of the centerline points
        
        # 1. find the length of the centerline, its current segments, 
        # and its segments if it were resampled at regular intervals  
        
        ds = [np.linalg.norm(cps[p+1]-cps[p]) for p in \
              list(range(len(cps)-1))]
        cum_d = [np.sum(ds[0:p+1]) for p in range(len(ds))]
        cum_d.insert(0,0.0)
        steps = np.linspace(0,np.sum(ds),N)
        
        segs = [] # segment of the resampled point
        percs = [] # relative distance along segment of resampled point
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
            plt.plot(np.array(cpsr)[:,1],np.array(cpsr)[:,0],'.',
                     color = (0,1,0))
            plt.axis('off')
            plt.show()
            
        # 4. smooth with a spline
        tck, u = interpolate.splprep([np.array(cpsr)[:,0], 
                                      np.array(cpsr)[:,1]], s = 3)
        u_new = np.linspace(0, 1, N)
        cpsrs = np.array(interpolate.splev(u_new, tck))
        
        if debug:
            plt.figure()
            plt.imshow(bw,cmap='gray')
            plt.title('Resampled Centerline Points')
            plt.plot(np.array(cps)[:,1],np.array(cps)[:,0],'r.')
            plt.plot(np.array(cpsr)[:,1],np.array(cpsr)[:,0],'g.')
            plt.plot(cpsrs[1],cpsrs[0],'b+')
            plt.axis('off')
            plt.show()
        
        centerline = np.fliplr(np.swapaxes(cpsrs,0,1))
        centerline = np.round(centerline,1)
    
    else:
        print('Error: centerline method not recognized')
    
    return centerline, curvature_end_1, curvature_end_2






def counter_clockwise(p1,p2,p3):
    '''Returns True if points p1, p2, and p3 form a counterclockwise
    triangle'''
    
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])



def intersect(p1,p2,p3,p4):
    '''Returns True if the line segment formed by p1 and p2 and the line
    segment formed by p3 and p4 intersect'''
    
    return counter_clockwise(p1,p3,p4) != \
        counter_clockwise(p2,p3,p4) \
        and counter_clockwise(p1,p2,p3) != \
        counter_clockwise(p1,p2,p4)



def self_intersect(centerline):
    '''Returns True if <centerline> crosses itself'''
    
    self_intersecting = False
    for s1 in range(len(centerline[0])-1):
        s2s = np.arange(len(centerline[0])-1).astype(np.uint8)
        s2s = np.delete(s2s,np.where(np.logical_and(s2s >= s1-1,
                                                    s2s <= s1+1))[0])
        for s2 in s2s:
            p1 = centerline[:,s1]; p2 = centerline[:,s1+1]
            p3 = centerline[:,s2]; p4 = centerline[:,s2+1]
            if intersect(p1,p2,p3,p4):
                self_intersecting = True; break
        if self_intersecting:
            break
    
    return self_intersecting


def unwrap(angles):
    '''Unwraps angles preventing extreme values'''
    
    for a in range(len(angles)):
        if angles[a] > np.pi:
            angles[a]  = -(2*np.pi-angles[a] )
        elif angles[a] < -np.pi:
            angles[a] = 2*np.pi+angles[a]
    
    return angles



def flag_bad_centerline(cline, max_length, max_angle):
    '''Flags a centerline if it is too long, too kinky, or crosses 
    itself'''
    
    flag = 0
    
    # determine if the centerline is too long
    length = 0
    for s in range(np.shape(cline)[0]-1):
        length += np.linalg.norm(cline[s+1]-cline[s])
    if length > max_length:
        flag = 1
    
    # determine if the centerline has any kinks
    if flag == 0:
        dxs = np.diff(cline[:,0],n=1,axis=0)
        dys = np.diff(cline[:,1],n=1,axis=0)
        angles = np.arctan2(dys,dxs)
        angles = np.diff(angles)
        angles = unwrap(angles)
        angles = np.degrees(angles)
        if np.max(np.abs(angles)) > max_angle:
            flag = 1
    
    # if not already flagged, determine if the centerline crosses itself
    if flag == 0:
        if self_intersect(np.swapaxes(cline,0,1)):
            flag = 1
    
    return flag



def rectify_centerlines(centerlines, angles_end_1, angles_end_2):    
    '''This function attempts to point all the centerline in the same
    direction by matching the ends from frame to fram by distance'''
    
    def dist(p1,p2):
        d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        return d
    
    
    for w, clines in enumerate(centerlines):
        # match ends based on proximity
        for f in range(len(clines)-1):
            ds = (
                dist(clines[f][0][0].astype(np.float32),
                     clines[f+1][0][0].astype(np.float32)),
                dist(clines[f][0][0].astype(np.float32),
                     clines[f+1][0][-1].astype(np.float32)),
                dist(clines[f][0][-1].astype(np.float32),
                     clines[f+1][0][0].astype(np.float32)),
                dist(clines[f][0][-1].astype(np.float32),
                     clines[f+1][0][-1].astype(np.float32))
                )
            
            match = np.where(ds == np.min(ds))[0][0]
            if match == 0 or match == 3: # do nothing
                pass
            elif match == 1 or match == 2: # flip centerline at f+1
                clines[f+1][0] = np.flipud(clines[f+1][0])
                temp_1 = copy.copy(angles_end_1[w][f+1])
                temp_2 = copy.copy(angles_end_2[w][f+1])
                angles_end_1[w][f+1] = temp_2
                angles_end_2[w][f+1] = temp_1
    
    return centerlines, angles_end_1, angles_end_2