# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:49:46 2022

Gravitational_model fits a binary moving image of a worm to a binary target
image. The moving image is generated by fitting a spline through <N> defining
centerline points (<main_pts>) as well as two sides offset from the 
centerline and filling in the enclosed area. During fitting, the defining 
points are pulled toward non-occluded parts of the target image by a 
"gravitational force" and the moving image is updated accordingly. Fitting is
complete when the IoU reaches a plateu.


Usage:
    
    1) Initialize the model with a binary target image (object = 255) and a
    an array of centerline coordinates (numpy array of size (2,50), with x and
    y values).
    
    grav_mod = Gravitational_model(mov_cline, target_img)
    
    
    2) Run the fitting algorithm. boolean <show> displays plots of the fitting
    process and, at the end, a plot of IoU values, and <write_video> saves a
    video of the fitting process in <save_dir>. The fit centerline is returned
    as a numpy array.
    
    new_cline = grav_mod.fit(show, write_video, save_dir)


Known issues:
    
    -Angles sharper than <a_max> can form even with the angle check due to
    redistribution of the main points after each iteration. This can be
    overcome by fitting a spline during the angle checking step 
    (<strict_angles> set to True), but this causes the model to get stuck.
    
    -The model has no maximum length, and therefore could be subject to 
    spaghettification under some circumstances.
    
    -I explicitly delete variables at the end of some functions; this is 
    probably not necessary or helpful.
    
    -


@author: P.D.McClanahan
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy
import sys
import os
import pickle
from datetime import datetime
from scipy import interpolate
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage.morphology import distance_transform_edt as dist_trans


class Gravitational_model():
    
    # metaparameters
    G = 0.1 # gravitational constant
    mtq = 0.25 # multiplier for rotation due to torque
    mtr = 0.25 # multiplier for translation due to net gravity
    N = 9 # number of main (defining centerline) points
    n = 50 # number of centerline points after resampling
    w = 8 # (initial) body width
    exponent = 2.0 # for equation of gravity
    a_max = 0.3*np.pi # sharpest allowable angle between two main points
    use_net_force = True
    use_torque = True
    use_weighted_targ = True
    strict_angles = False # tends to cause the model to get stuck
    wt_scl = 0.2
    
    # set up width profile
    body_coords = np.array([0,.005,0.05,0.2,0.5,0.8,0.95,.995,1.0])
    body_rel_ws = np.array([0,4,6,7.4,8,7.4,6,4,0])/8
    spl = interpolate.splrep(body_coords, body_rel_ws,k=1,s=0)
    u = np.linspace(0, 1, n)
    w_factors = interpolate.splev(u, spl) # relative width at each point
    # plt.plot(body_coords, body_rel_ws, 'o', u, w_factors)
    # plt.show()
    
    del body_coords, body_rel_ws, spl, u
    
    
    def __init__(self, init_centerline, target_img):
        '''Initializes a gravitational worm model with the N model points
        distributed along <init_centerline> and the target image set as 
        <target image>'''
        self.target_img = target_img
        self.target_area = len(np.where(self.target_img == 255)[0])
        
        # find initial centerline points
        tck, u = interpolate.splprep([init_centerline[0], init_centerline[1]],
                                     s=0)
        u_new = np.linspace(0, 1, self.N)
        self.main_pts = np.array(interpolate.splev(u_new, tck))
        
        # draw moving image, adjust width, and draw moving image again
        self.draw_moving_img()
        self.update_width()
        self.draw_moving_img()
             
        
    def draw_moving_img(self):
        '''Draws an image based on the locations of the N main points, width,
        and w_factors (width factors)'''
        
        # reinterpolate centerline to n points
        tck, u = interpolate.splprep([self.main_pts[0], self.main_pts[1]],s=0)
        u_new = np.linspace(0, 1, self.n)
        mov_cl = np.array(interpolate.splev(u_new, tck))
        
        # find dorsal and ventral sides by offsetting from centerline
        dx = np.diff(mov_cl[0]); dy = np.diff(mov_cl[1])
        dx = np.mean(np.vstack((dx[:-1],dx[1:])),0)
        dy = np.mean(np.vstack((dy[:-1],dy[1:])),0)
        angles = np.arctan2(dy,dx) # gives n-1, need n-2
        body_ws = self.w*self.w_factors
        
        v_pts = copy.copy(mov_cl)        
        v_pts[0,1:-1] = v_pts[0,1:-1] + body_ws[1:-1]*np.sin(angles)
        v_pts[1,1:-1] = v_pts[1,1:-1] - body_ws[1:-1]*np.cos(angles)
                
        d_pts = copy.copy(mov_cl)        
        d_pts[0,1:-1] = d_pts[0,1:-1] - body_ws[1:-1]*np.sin(angles)
        d_pts[1,1:-1] = d_pts[1,1:-1] + body_ws[1:-1]*np.cos(angles)
        
        
        # # fill in body (this version fills self-overlaping areas)
        # self.moving_img = np.zeros(np.shape(self.target_img))
        # for p in range(len(mov_cl[0])-1):
        #     xs = np.array([d_pts[0][p],d_pts[0][p+1],v_pts[0][p+1],
        #                    v_pts[0][p]],dtype = 'int32')
        #     ys = np.array([d_pts[1][p],d_pts[1][p+1],v_pts[1][p+1],
        #                    v_pts[1][p]],dtype = 'int32')
        #     contours = np.flipud(np.rot90(np.vstack((xs,ys))))
        #     cv2.fillPoly(self.moving_img, pts = [contours], color=255)
        
        
        # fill in body (this version does not fill self-overlaping areas)
        self.moving_img = np.zeros(np.shape(self.target_img))
        xs2 = np.concatenate((d_pts[0],np.flip(v_pts[0])))
        ys2 = np.concatenate((d_pts[1],np.flip(v_pts[1])))
        contours2 = np.flipud(np.rot90(np.vstack((xs2,ys2))))
        contours2 = contours2.astype(np.int32)
        cv2.fillPoly(self.moving_img, pts = [contours2], color=255)
    
    
    def calc_Fg(self):
        '''Calculates the sum of the gravitational forces exerted on each 
        point defining the moving model's location by the unoccluded pixels of
        the target image'''
 
        # create array of non-occluded target image points
        occl_targ_img = copy.copy(self.target_img)
        occl_targ_img[np.where(self.moving_img == 255)] = 0
        occl_targ_pts = np.array(np.where(occl_targ_img == 255))
        occl_targ_pts = np.flipud(occl_targ_pts)
        
        if self.use_weighted_targ:
            rev_moving_img = 255*np.ones(np.shape(self.moving_img))
            rev_moving_img[np.where(self.moving_img  == 255)] = 0
            occl_targ_wt_img = dist_trans(rev_moving_img)
            occl_targ_wts = occl_targ_wt_img[np.where(occl_targ_img == 255)]
            occl_targ_wts = occl_targ_wts * self.wt_scl + (1-self.wt_scl)
        
        # calculate the summation of forces on each point N
        self.forces = np.empty((self.N,2)) # x, y
        for mp in range(self.N):
            gx_mp = 0
            gy_mp = 0
            for p in range(len(occl_targ_pts[0])):
                xy = occl_targ_pts[:,p]-self.main_pts[:,mp]
                dist = np.linalg.norm(xy)
                if dist < .75: # prevent extreme forces at close proximities
                    dist = .75
                if self.use_weighted_targ:
                    mag = self.G * (1/dist**self.exponent) * occl_targ_wts[p]
                else:
                    mag = self.G * (1/dist**self.exponent)
                gx_mp += mag*xy[0]
                gy_mp += mag*xy[1]
            
            self.forces[mp,0] = gx_mp
            self.forces[mp,1] = gy_mp
        
        
        if self.mtq > 0 and self.mtr > 0:
            # calculate net force
            self.net_force = np.sum(self.forces,0)
            
            # calculate torque
            self.cm = np.mean(self.main_pts,1)
            self.torque = 0
            for mp in range(self.N):
                moment_arm = self.main_pts[:,mp] - self.cm
                self.torque += np.cross(moment_arm,self.forces[mp])
        
        # try: # p will not be defined if there is total occlusion
        #     del occl_targ_img, occl_targ_pts, mp, gx_mp, gy_mp, p, dist, mag
        # except:
        #     pass
        
    
    def update_pts(self):
        '''Updates the main model points one by one according to the
        gravitation forces on them, but only if the maximum permissable angle
        formed by three of them is not exceeded'''
        
        # move the spline point according in the direction of gravity
        # DEBUGGING
        #if self.it == 58: import pdb; pdb.set_trace()
        # inds = np.linspace(0, self.N-3, self.N-2).astype(np.uint8)
        # aa = []
        # for i in inds:
        #     aa.append(self.find_angle(self.main_pts[:,i],self.main_pts[:,i+1],
        #                              self.main_pts[:,i+2]))
        # old_pts = copy.copy(self.main_pts)
        
        
        # apply forces to main points
        for mp in range(self.N):
            new_pts = copy.copy(self.main_pts)
            # if self.it == 58 and mp == 8: import pdb; pdb.set_trace()
            
            new_pts[:,mp] = new_pts[:,mp] + self.forces[mp]
            
            # resample spline and check all angles and update if angles are
            # within limits (not resampling can allow sharp angles to form)
            if self.strict_angles:
                tck, u = interpolate.splprep([new_pts[0], new_pts[1]], s=0)
                u_new = np.linspace(0, 1, self.N)
                new_pts_res = np.array(interpolate.splev(u_new, tck))
                
                inds = np.linspace(0,self.N-2,1).astype(np.uint8)
                fine = True
                for i in inds:
                    a = self.find_angle(new_pts_res[:,i],new_pts_res[:,i+1],
                                         new_pts_res[:,i+2])
                    if abs(a) > self.a_max:
                        fine = False
                        
                if fine:
                    self.main_pts[:,mp] = new_pts[:,mp]
            
            else:
                inds = np.linspace(mp-2,mp,3).astype(np.uint8)
                inds = np.delete(inds, np.where(inds >= self.N-2)[0])
                # inds = np.linspace(0, self.N-3, self.N-2).astype(np.uint8)
                fine = True
                for i in inds:
                    a = self.find_angle(new_pts[:,i],new_pts[:,i+1],
                                         new_pts[:,i+2])
                    if abs(a) > self.a_max:
                        fine = False
                        
                if fine:
                    self.main_pts[:,mp] = new_pts[:,mp]
        
        # apply torque and net force
        if self.mtq > 0 and self.mtr > 0:
            for mp in range(self.N):
                self.main_pts[:,mp] = self.rotate(self.main_pts[:,mp],
                                                  self.cm, 
                                                  self.mtq*self.torque)
                
                self.main_pts[:,mp] = self.main_pts[:,mp] + \
                                      self.mtr*self.net_force


        # redistribute main model points along centerline
        tck, u = interpolate.splprep([self.main_pts[0], self.main_pts[1]],
                                     s=0)
        u_new = np.linspace(0, 1, self.N)
        self.main_pts = np.array(interpolate.splev(u_new, tck))
        
    
    def update_width(self):
        '''Adjusts the width <self.w> such that the area of the moving model
        will be approximately equal to the area of the target'''
        
        moving_area = len(np.where(self.moving_img == 255)[0])
        factor = self.target_area/moving_area
        self.w = self.w * factor


    def show_IoU(self, IoU, show_it = 0, ret_image = False):
        '''Calculates the loss and creates a visual showing an overlay of the
        moving and target images'''
        #import pdb; pdb.set_trace()
        im1 = np.uint16(self.moving_img); im2 = np.int16(self.target_img)
        comb_img = (np.uint16(self.moving_img)+np.uint16(self.target_img))/2
        
        fig,ax = plt.subplots()
        plt.tight_layout()
        plt.imshow(comb_img,cmap = 'gray',aspect='equal')
        ax.axis('off')
        #fig.set_frameon(False)
        plt.text(5*self.plt_scl,20*self.plt_scl,'IoU = '+str(round(IoU,2)),
                 color = 'white')
        exagg = 10
        
        for p in range(self.N):
            x = np.array([self.main_pts[0,p],self.main_pts[0,p]+ \
                          self.forces[p,0]*exagg])
            y = np.array([self.main_pts[1,p],self.main_pts[1,p]+ \
                          self.forces[p,1]*exagg])
            plt.arrow(x[0],y[0],x[1]-x[0],y[1]-y[0],color='r',
                      width = .5 * self.plt_scl,
                      head_width = 5 * self.plt_scl)
            plt.plot(x[0],y[0],'r.')
        
        if show_it:
            plt.text(5*self.plt_scl, 40*self.plt_scl, 
                     'Iter = '+str(round(self.it)),color = 'white')
        
        
        # image for demo video    
        if ret_image:    
            canvas = FigureCanvas(fig) # for demo vid
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), 
                                  dtype='uint8').reshape(int(height),
                                  int(width), 3)
        plt.show()
        if ret_image:
            return image

    
    def fit(self, show, write_video, save_dir):
        
        if write_video and not show:
            print('WARNING: Write_video without show not supported')
            
        if show:
            self.plt_scl = self.target_img.shape[0]/400 
        
        plateau = False
        IoUs = []
        main_points = []
        if write_video:
            stack = []
        self.it = 0
        
        while not plateau:
            self.calc_Fg()
            self.update_pts()
            self.update_width()
            self.draw_moving_img()
            
            IoUs.append(self.calc_IoU(self.moving_img, self.target_img))
            main_points.append(self.main_pts)
            
            if show and not write_video:
                print(self.it)
                self.show_IoU(IoUs[-1],True)
            
            elif write_video:
                print(self.it)
                stack.append(self.show_IoU(IoUs[-1],True,True))
                        
            # check for doneness       
            if self.it > 50 and np.sum(np.diff(IoUs[-50:])<0) >= 25 and \
                IoUs[-1] > 0.6 or self.it > 300:
                plateau = True
                best_it = np.where(IoUs == np.max(IoUs))[0][0]
                best_main_pts = main_points[best_it]
                
                # create an <n> point centerline along the main model points
                tck, u = interpolate.splprep([best_main_pts[0], 
                                              best_main_pts[1]], s=0)
                u_new = np.linspace(0, 1, self.n)
                new_centerline = np.array(interpolate.splev(u_new, tck))
                
                if show:
                    plt.plot(IoUs,'k-')
                    plt.plot(best_it,np.max(IoUs),'r+',markersize = 15)
                    plt.xlabel('Iteration')
                    plt.ylabel('IoU')
                    plt.show()
             
            # adjust equation of gravity
            if self.it == 25:
                self.exponent = 2.5
                self.G = 0.05
                self.mtq = 0.125
                self.mtr = 0.125
                self.wt_scl = 0.4
            if self.it > 50 and IoUs[-1] > 0.7:
                self.exponent = 3.0
                self.G = 0.1
                self.mtq = 0.0
                self.mtr = 0.0
                self.wt_scl = 1
            
            self.it+=1
        
        if write_video:
            self.write_demo_vid(save_dir, stack)
        
        return new_centerline
     
        
    @staticmethod
    def rotate(p, rot_center, angle):
        """Rotates a point <p> counterclockwise about point <rot_center> by
        <angle> (degrees)"""
        
        angle = np.radians(angle)
        rcx, rcy = rot_center
        px, py = p
    
        x = rcx + np.cos(angle) * (px - rcx) - np.sin(angle) * (py - rcy)
        y = rcy + np.sin(angle) * (px - rcx) + np.cos(angle) * (py - rcy)
        
        return x, y
    
    
    @staticmethod
    def find_angle(p1, p2, p3):
        '''Finds the two-dimensional angle between three points <p1>, <p2>,
        and <p3>. Angles to the left are positive.'''
        
        # calculate the angles of the two segments
        dxs = np.diff(np.array((p1[0],p2[0],p3[0])),n=1,axis=0)
        dys = np.diff(np.array((p1[1],p2[1],p3[1])),n=1,axis=0)
        angles = np.arctan2(dys,dxs) # range: (-pi,pi]
        
        # subtract those two angles and unwrap if necessary
        diff_angle = np.diff(angles,n=1,axis = 0) # right turns are negative
        angle = copy.copy(diff_angle)
        if angle > np.pi:
            angle  = -(2*np.pi-angle)
        elif angle < -np.pi:
            angle = 2*np.pi+angle
            
        return angle
    
    
    @staticmethod
    def calc_IoU(img_1, img_2):
        '''Calculates the Intersection over Union between <img_1> and <img_2>, 
        defined as the area of overlap (intersection) divided by the total 
        coverage (union)'''
        
        comb_img = img_1 + img_2        
        inter = np.sum(np.where(comb_img == 510)[0])
        union = np.sum(np.where(comb_img == 255)[0]) + inter
        IoU = inter / union
        return IoU
    
    
    @staticmethod
    def write_demo_vid(save_dir, stack):
        '''Uses the frames in <stack> to write a color video in <save_dir> 
        with a filename containing a timestamp'''
        
        vid_file = datetime.now().strftime("%Y%m%d%H%M%S") + \
                   '_gravitational_model_demo.avi'
        
        out_w = np.shape(stack[0])[1]
        out_h = np.shape(stack[0])[0]
        v_out = cv2.VideoWriter(save_dir+'\\'+vid_file,
                cv2.VideoWriter_fourcc('M','J','P','G'),
                10, (out_w,out_h), 1)
        
        for f in range(len(stack)):
            img = cv2.cvtColor(np.squeeze(stack[f]),cv2.COLOR_RGB2BGR)
            v_out.write(img)
            
        v_out.release()
    
    
##############################################################################

  
# testing
if __name__ == '__main__':
    try:
        import time
        
        show = True
        write_video = True
        save_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nic'+\
            r'tation\test_output'
        
        
        # # LOAD LOOPED WORM
        # init_cond_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\c'+\
        #     r'ode\nictation\test_files\20220215_ext_debugging_start.p'
        # with open(init_cond_file, 'rb') as f:
        #     mov_cline, target_img = pickle.load(f)
        
        # # FIT TO LOOPED WORM
        # start_time = time.time()
        # grav_mod = Gravitational_model(mov_cline, target_img)
        # new_cline = grav_mod.fit(show, write_video, save_dir)
        # elapsed = time.time()-start_time
        # print(str(elapsed) +' seconds elapsed')
        
        
        # # FIT TO A BOX (uses initial condition from above)
        # target_img = np.zeros((400,400),dtype = np.uint8)
        # target_img[100:150:,-150:-100] = 255
        # grav_mod = Gravitational_model(mov_cline, target_img)
        # new_cline = grav_mod.fit(show, write_video, save_dir)
        
        
        # EXPERIMENT WITH SCALING AND FIT TIME
        # start_time = time.time()
        
        # scale_prop = 0.25
        # width = int(target_img.shape[1] * scale_prop)
        # height = int(target_img.shape[0] * scale_prop)
        # dim = (width, height)
        # target_img_scaled = cv2.resize(
        #    target_img, dim, interpolation = cv2.INTER_AREA)
        # thr, target_img_scaled = cv2.threshold(target_img_scaled,127,255,
        #    cv2.THRESH_BINARY)
        
        # mov_cline_scaled = mov_cline * scale_prop
        
        # grav_mod = Gravitational_model(mov_cline_scaled, target_img_scaled)
        # new_cline = grav_mod.fit(show, write_video, save_dir)
        # elapsed = time.time()-start_time
        # print(str(elapsed) +' seconds elapsed')
        
        # LOAD A 'HIGH DISPLACEMENT' FITTING EXAMPLE
        init_cond_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\cod'+\
            r'e\nictation\test_files\20220312_gravmod_debugging\init_cond.p'
        with open(init_cond_file, 'rb') as f:
            mov_cline, target_img = pickle.load(f)
        
        start_time = time.time()
  
        grav_mod = Gravitational_model(mov_cline, target_img)
        new_cline = grav_mod.fit(show, write_video, save_dir)
        elapsed = time.time()-start_time
        print(str(elapsed) +' seconds elapsed')
        
        
        
        pass
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
