# -*- coding: utf-8 -*-
"""
This script creates a random deformable worm whose shape is determined by 
eigenworm coefficients, and matches it to a real worm segmentation by gradient
descent.

Issues
-deloopify calculation of joint angles in update_bw_image
-clean up extension optimization, right now it can go outside model bounds
-automatically adapt to centerlines of different lengths and numbers of
 segments

Created on Sat Dec  4 17:22:07 2021
@author: P. D. McClanahan (pdmcclanahan@gmail.com)
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



class Eigenworm_model():
    # x, y, rot, stretch, eigenworms 1 thru 5; (low, high, repeating?)
    # limits based on where worm makes self-contact
    # parameter_limits = [(150,250,0),(150,250,0),(0,360,1),(0.25,1.1,0),
    #                     (0,16.5,0),(0,14,0),(0,15,0),(0,12,0),(0,14,0)]
    # parameter_limits = [(150,250,0),(150,250,0),(0,360,1),(0.25,1.1,0),
    #                     (-8,8.25,0),(-7,7,0),(-7.5,7.5,0),(-6,6,0),(-7,7,0)]
    parameter_limits = [(150,250,0),(150,250,0),(0,360,1),(0.1,1.1,0),
                        (-15,15,0),(-15,15,0),(-15,15,0),(-15,15,0),(-15,15,0)]
    
    eigenworm_file = os.path.split(__file__)[0] + '\\test_files\\20211212_Sc_eigenworms_50.p'
    #eigenworm_file = os.path.split(__file__)[0] + '\\test_files\\20211212_Sc_eigenworms.p'

    
    # number of points / angles -1 in a centerline
    N = 50
    
    # number of eigenworms to use
    n_coeff = len(parameter_limits) - 4
    
    # dimensions of a worm in pixels
    base_length = 253
    #base_length = 126
    body_coords = np.array([0,.03,0.1,0.5,0.9,.97,1.0])
    body_coords = np.array([0,0.015,.03,0.1,0.5,0.9,.97,.985,1.0])
    body_widths = np.array([0,5,6,8,6,5,0])
    body_widths = np.array([0,5,7,8,9,8,7,5,0])
    
    # linear interpolation of width along the body
    width_factors = np.empty(N)
    for s in range(N):
        body_coord = s/N
        pt_1 = np.max(np.where(body_coord >= body_coords))
        pt_2 = pt_1 + 1
        
        slope = (body_widths[pt_2] - body_widths[pt_1]) / \
            (body_coords[pt_2] - body_coords[pt_1])
        intercept = body_widths[pt_1] - slope * body_coords[pt_1]
        width_factors[s] = slope*(body_coord) + intercept
    del body_coords, body_widths, s, body_coord, pt_1, pt_2, slope, intercept
    
    
    def __init__(self):
        """Initialize a model worm"""
        
        with open(self.eigenworm_file, 'rb') as f:
            eigendict = pickle.load(f)
            self.EVecs = eigendict['eigenvectors'][:,0:self.n_coeff]
        
        # # adjust eigenvector length to match N-1, the number of angles
        # if self.N-1 != len(self.EVecs[:,0]):
        #     EVecs_new = np.empty((self.N-1,len(self.EVecs[0])),np.float32)
        #     for i in range(len(self.EVecs[0])):
        #         tck, u = interpolate.splprep([np.linspace(0,1,len(self.EVecs[:,i])),self.EVecs[:,i]], s=0)
        #         unew = np.linspace(0,1,self.N-1)
        #         EVecs_new[:,i] = np.array(interpolate.splev(unew,tck))[1]
        #     self.EVecs = EVecs_new                               
            

        
    def randomize_parameters(self):
        """Sets the model parameters to random values within their limits and
        creates a centerline and bw image based on those parameters"""
        
        self.parameters = list()
        counter = 0
        for pl in self.parameter_limits:
            if counter < 3:
                self.parameters.append(random.uniform(pl[0],pl[1]))
            elif counter == 3:
                 self.parameters.append(random.uniform(0.75,pl[1]))
            else:
                self.parameters.append(random.uniform(pl[0],pl[1]/5))
            counter += 1
        self.parameters_to_centerline()
        self.update_bw_image()
    
    
    
    def set_centerline(self,centerline):
        """Fits the model parameters to the centerline and creates a bw image
        of the worm"""
        self.N = np.shape(centerline[0])[0]
        self.centerline = np.array(centerline,np.float32)
        self.centerline_to_parameters()
        self.update_bw_image()
    
    
    
    def set_parameters(self,params):
        """Creates a centerline and a bw image based on the params"""
    
        self.parameters = params
        self.parameters_to_centerline()
        self.update_bw_image()
        
        
        
    def centerline_to_parameters(self):
        """Finds a set of model parameters that closely match the given 
        centerline coordinates"""
        
        # find the midpoint coordinates
        
        x = self.centerline[0][int(self.N/2)]
        y = self.centerline[1][int(self.N/2)]
        
        
        # turn centerline into angles and find the rotation angle            
        dxs = np.diff(self.centerline[0],n=1,axis=0)
        dys = np.diff(self.centerline[1],n=1,axis=0)
        angles = np.degrees(np.unwrap(np.arctan2(dys,dxs)))
        rot = np.mean(angles) + 360
        self.angles = angles - rot
        ####
        angles = angles - np.mean(angles)
        # work okangles = angles -angles[0]
        # angles = angles - np.mean([angles[24],angles[25]])
        # leaving angles as is or using self.rot fails
        
        # turn angles into eigenworm coefficient parameters
        A = self.EVecs[:,0:self.n_coeff]
        ####
        X = np.linalg.lstsq(A,np.radians(angles),rcond=None)
        self.coeffs = X[0]
        
        
        # find the total length and calculate the stretch coefficient
        length = np.sum(np.sqrt(np.square(np.diff(self.centerline[1]))+np.square(np.diff(self.centerline[0]))))
        stretch = length/self.base_length
        
        
        # set parameters
        self.parameters = [x,y,rot,stretch, X[0][0],X[0][1],X[0][2],X[0][3],X[0][4]]
        

    
    
    
    def parameters_to_centerline(self):
        """Uses the model parameters to calculate the centerline in Euclidian
        space. N is the number of centerline points"""
        
        x_shift = self.parameters[0]
        y_shift = self.parameters[1]
        rot = self.parameters[2]
        stretch = self.parameters[3]
        coeffs = self.parameters[4:]
        N = self.N
        seg_length = stretch*(self.base_length / N)
        
        
        # calculate centerline angles from eigenworm coefficients
        self.angles = np.zeros(N-1)
        for i in range(len(coeffs)):
            self.angles = self.angles + coeffs[i]*self.EVecs[:,i]
            
        
        # calculate points from angles
        X = np.empty((np.shape(self.angles)[0]+1)); X[0] = 0
        Y = np.empty((np.shape(self.angles)[0]+1)); Y[0] = 0
        for p in range(np.shape(X)[0]-1):
            X[p+1] = X[p] + seg_length * np.cos(self.angles[p])
            Y[p+1] = Y[p] + seg_length * np.sin(self.angles[p])
        
        
        # center the centerline on origin
        # X = X - np.mean(X); Y = Y-np.mean(Y);
        X = X - X[int(N/2)]; Y = Y - Y[int(N/2)]
        
        
        # rotate points
        rot_center = [X[int(N/2)],Y[int(N/2)]]
        for i in range(len(X)):
            X[i],Y[i] = self.rotate(rot_center,[X[i],Y[i]],rot)
            
        
        # shift points
        X = X + x_shift
        Y = Y + y_shift
        
        self.centerline = np.array([X,Y])
        
    

    def update_bw_image(self):
        """Makes a binary image of a worm based on the model parameters"""
        
        # create blank image
        self.bw_image = np.zeros((400,400),dtype = 'uint8')
        
        
        # calculate joint angles along centerline
        dy = np.diff(self.centerline[1])
        dx = np.diff(self.centerline[0])
        segment_angles = np.arctan2(dy,dx)
        segment_angles = np.unwrap(segment_angles)
        joint_angles = np.convolve(segment_angles, np.ones(2), 'valid') / 2
        
        
        # calculate perimeter points on the left and right hand side of the
        # centerline
        RHS_x = self.centerline[0][1:-1] + self.width_factors[1:-1] * np.sin(joint_angles)
        RHS_y = self.centerline[1][1:-1] - self.width_factors[1:-1] * np.cos(joint_angles)
        
        LHS_x = self.centerline[0][1:-1] - self.width_factors[1:-1] * np.sin(joint_angles)
        LHS_y = self.centerline[1][1:-1] + self.width_factors[1:-1] * np.cos(joint_angles)
        
        # create a list of perimeter points
        end_1 = np.expand_dims(np.array([self.centerline[0,0],self.centerline[1,0]]),0)
        end_2 = np.expand_dims(np.array([self.centerline[0,-1],self.centerline[1,-1]]),0)
        RHS = np.swapaxes(np.vstack((np.array(RHS_x),np.array(RHS_y))),1,0)
        LHS = np.flip(np.swapaxes(np.vstack((np.array(LHS_x),np.array(LHS_y))),1,0),0)
        perimeter = np.vstack((end_1,RHS,end_2,LHS))
        perimeter = perimeter.astype('int32')

        
        # fill the contour on a BW image
        cv2.fillPoly(self.bw_image, [perimeter],255)
    
    def show_bw_image(self):
        """"Shows the current bw image of the model"""
        #import pdb; pdb.set_trace()
        fig,ax = plt.subplots()
        ax.imshow(self.bw_image, cmap = 'gray')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()

    
    def extend_in_track(self, direction = 1, amount = 0.01):
        """Expands or contracts the worm by <amount> in <direction> specified,
        working directly on the centerline
        """
        
        # copy parameters to restore current position later
        self.pre_expansion_parameters = copy.deepcopy(self.parameters)
        
        # fit a parametric spline to the current centerline
        tck, u = interpolate.splprep([self.centerline[0], self.centerline[1]], s=0)
        
        # use the spline to interpolate new extended centerline points
        if direction == 1:
            unew = np.linspace(0, 1+amount, self.N)
        elif direction == -1:
            unew = np.linspace(0-amount, 1, self.N)
        self.centerline = np.array(interpolate.splev(unew, tck))
        
        # fit the model parameters to the new centerline
        self.centerline_to_parameters()
        self.curb_parameters()
        # self.update_bw_image()
        
        # # recalculate the centerline and bw image to reflect the fitted
        # # parameters
        self.parameters_to_centerline()
        self.update_bw_image()

    
    def undo_last_extension(self):
        """Restores the parameters prior to the last call of expand_in_track
        """
        self.parameters = self.pre_expansion_parameters
        self.parameters_to_centerline()
        self.update_bw_image()


    def curb_parameters(self):
        """Restores parameters back inside their permissable limits"""
        for i,p in enumerate(self.parameters):
            
            if self.parameter_limits[i][2] == 0: # non repeating parameter range
                
                if p < self.parameter_limits[i][0]:
                   p = self.parameter_limits[i][0]
                elif p > self.parameter_limits[i][1]:
                    p = self.parameter_limits[i][1]
            
            else: # repeating or cyclic parameter range
                
                if p < self.parameter_limits[i][0]:
                    p = self.parameter_limits[i][1] - p 
                elif p > self.parameter_limits[i][1]:
                    p = self.parameter_limits[i][0] + p - self.parameter_limits[i][1]
                


    @staticmethod    
    def rotate(rot_center, p, angle):
        """Rotates a point counterclockwise around a point by an angle
        (degrees)"""
        angle = np.radians(angle)
        rcx, rcy = rot_center
        px, py = p
    
        x = rcx + np.cos(angle) * (px - rcx) - np.sin(angle) * (py - rcy)
        y = rcy + np.sin(angle) * (px - rcx) + np.cos(angle) * (py - rcy)
        return x, y




class Gradient_descent():


    def __init__(self, model, target_img, n_iter, lr, grad_steps, max_steps, save_dir, show = 1, demo_vid = 0):
        self.model = model # deformable worm model
        self.moving_img = model.bw_image # image produced by deformable model
        self.target_img = target_img # worm image to be matched
        self.n_iter = n_iter # max number of iterations
        self.lr = lr # learning rate
        self.grad_steps = grad_steps # step size used to calculate the gradient
        self.max_steps = max_steps # maximum allowed change in one iteration
        self.show = show # plot registration in real time
        self.demo_vid = demo_vid # write a video of registration
        self.save_dir =  save_dir
        
    
    def run(self):
        '''Runs the gradient descent according to the supplied parameters'''
        
        
        # run the optimization loop until target IoU is reached or n_iter is
        # exceeded
        losses = []
        for it in range(self.n_iter):
            # gradient decent with model parameters
            self.calc_grad()
            self.update_parameters()

            # extend / contract worm ends
            if it > 50 and it%5 == 0:
                # import pdb; pdb.set_trace(); sf = 0.05
                stretch_factors = [-0.01,0.01]
                losses_1 = np.zeros(len(stretch_factors)); losses_2 = copy.copy(losses_1)
                #baseline_loss = self.calc_loss('TO')
                baseline_loss = self.calc_loss()
                
                # self.model.extend_in_track(direction = 1, amount = 0)
                # baseline_loss = self.calc_loss()
                # self.model.undo_last_extension()
                
                for i, sf in enumerate(stretch_factors):
                    self.model.extend_in_track(direction = 1, amount = sf)
                    # losses_1[i] = self.calc_loss('TO')
                    losses_1[i] = self.calc_loss()
                    self.model.undo_last_extension()
                    
                    self.model.extend_in_track(direction = -1, amount = sf)
                    #losses_2[i] = self.calc_loss('TO')
                    losses_2[i] = self.calc_loss()
                    self.model.undo_last_extension()
                
                if np.min(losses_1) < baseline_loss:
                    sf = stretch_factors[np.where(losses_1 == np.min(losses_1))[0][0]]
                    self.model.extend_in_track(direction = 1, amount = sf)
                    
                if np.min(losses_2) < baseline_loss:
                    sf = stretch_factors[np.where(losses_2 == np.min(losses_2))[0][0]]
                    self.model.extend_in_track(direction = -1, amount = sf)
            
            # calculate loss and display current overlap
            IoU, image = self.show_IoU(it)
            losses.append(1-IoU)
            
            # check loss for learning rate reduction criteria
            # (check for oscillations and either reduce learning rate or stop)
            # if IoU > 0.5 and slow_downs > 0:
            #     learn_rate = learn_rate/10
            #     slow_downs -= 1
            if IoU > 0.95:
                break
            
            # add current overlap to demo video stack if applicable
            if self.demo_vid == True:
                if it == 0:
                    stack = image
                elif it == 1:
                    stack = np.stack((stack,image),-1)
                else:
                    stack = np.concatenate((stack,np.expand_dims(image,-1)),-1)
        if self.demo_vid:
            self.write_demo_vid(stack)
        
        return losses
     
    def calc_grad(self):
        '''Calculates the gradient of each model parameter using the grad_step
        provided'''
        
        
        #import pdb; pdb.set_trace()
        self.gradient = np.zeros(len(self.model.parameters))
        
        for i in range(len(self.gradient)):
            self.model.parameters[i] -= self.grad_steps[i]
            self.model.set_parameters(self.model.parameters)
            loss1 = self.calc_loss()
            self.model.parameters[i] += 2*self.grad_steps[i]
            self.model.set_parameters(self.model.parameters)
            loss2 = self.calc_loss()
            self.model.parameters[i] -= self.grad_steps[i]
            self.gradient[i] = (loss2-loss1)/(2*self.grad_steps[i])
            self.model.set_parameters(self.model.parameters)
        

    def calc_loss(self, loss_type = 'IoU'):
        '''Calculates the loss type specified based on the moving and target
        images'''
        
        self.moving_img = self.model.bw_image
        im1 = np.uint16(self.moving_img); im2 = np.int16(self.target_img)
        im_comb = im1+im2
        intersection = np.sum(np.where(im_comb == 510)[0])
        
        if loss_type == 'IoU':
            union = np.sum(np.where(im_comb == 255)[0]) + intersection
            IoU = intersection / union
            loss = 1- IoU
        
        elif loss_type == 'TO': # total overlap            
            target_area = np.sum(np.where(im2 == 255)[0])
            TO = intersection / target_area
            loss = 1 - TO

        return loss
    
    
    
    def update_parameters(self):
        '''Updates the model parameters based on the gradient and learning
        rate'''
        
        # calculate new parameters
        diff =  -(self.lr * self.gradient)
        for p in range(len(diff)):
            if diff[p] > self.max_steps[p]:
                diff[p] = self.max_steps[p]
            elif diff[p] < -self.max_steps[p]:
                diff[p] = -self.max_steps[p]
        new_parameters = self.model.parameters + diff
        
        
        # check that new parameters are in range and adjust if necessary (could move this into the model)
        for x in range(len(new_parameters)):
            if self.model.parameter_limits[x][2] == 0: # non repeating parameter range
                if new_parameters[x] < self.model.parameter_limits[x][0]:
                    new_parameters[x] = self.model.parameter_limits[x][0]
                elif new_parameters[x] > self.model.parameter_limits[x][1]:
                    new_parameters[x] = self.model.parameter_limits[x][1]
            else: # repeating or cyclic parameter range
                if new_parameters[x] < self.model.parameter_limits[x][0]:
                    new_parameters[x] = self.model.parameter_limits[x][1] - new_parameters[x] 
                elif new_parameters[x] > self.model.parameter_limits[x][1]:
                    new_parameters[x] = self.model.parameter_limits[x][0] + new_parameters[x] - self.model.parameter_limits[x][1]
                
        # update parameters
        self.model.set_parameters(new_parameters)
    
    
    def show_IoU(self, iteration = -1):
        '''Calculates the loss and creates a visual showing an overlay of the
        moving and target images'''
        
        loss = self.calc_loss(); IoU = 1-loss
        
        im1 = np.uint16(self.moving_img); im2 = np.int16(self.target_img)
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
        
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        
        return IoU, image



    def write_demo_vid(self, stack):
        
        vid_file = datetime.now().strftime("%Y%m%d%H%M%S") +'_grad_descent_eigenworm_demo.avi'
        
        out_w = np.shape(stack)[1]
        out_h = np.shape(stack)[0]
        v_out = cv2.VideoWriter(self.save_dir+'\\'+vid_file,
                cv2.VideoWriter_fourcc('M','J','P','G'),
                10, (out_w,out_h), 0)
        
        for f in range(np.shape(stack)[2]):
            v_out.write(np.squeeze(stack[:,:,f]))
            
        v_out.release()
    



def main():
    try:
        
        # # set centerline and parameters
        # parameters = np.array([199.64189211, 200.33110024, 270.62147052, 1.1,
        #           7.54171994, 3.32793145, -0.59873046, -4.53026587,
        #         -0.27242353])
        # centerline = np.array([[107.39083612, 106.3878604 , 105.65657258, 105.35655172,
        #         105.63693042, 106.7110269 , 108.84108073, 112.24248961,
        #         116.88658546, 122.24751018, 127.81301377, 133.26606232,
        #         138.49443453, 143.58256135, 148.6832512 , 153.88915394,
        #         159.25616715, 164.7823699 , 170.34221861, 175.77930401,
        #         180.92621694, 185.66871693, 189.92256528, 193.67041411,
        #         196.91114687, 199.64189211, 201.8134617 , 203.36940756,
        #         204.23630938, 204.37264127, 203.80665141, 202.60060049,
        #         200.83047929, 198.52053334, 195.64069884, 192.16304203,
        #         188.13249226, 183.64021986, 178.81499274, 173.68812297,
        #         168.27399727, 162.70835278, 157.23003544, 151.91329859,
        #         146.80488073, 141.97427545, 137.48440123, 133.42143592,
        #         130.39413647, 129.25964398],
        #         [254.03662972, 248.56174199, 243.04399117, 237.48608298,
        #         231.92714928, 226.46576917, 221.32347137, 216.91770542,
        #         213.84968819, 212.35274389, 212.2784081 , 213.39403234,
        #         215.30308447, 217.55948015, 219.78733092, 221.75683153,
        #         223.23179772, 223.89620657, 223.63459928, 222.44360882,
        #         220.32473096, 217.41113356, 213.82154721, 209.70645484,
        #         205.18119162, 200.33110024, 195.20619619, 189.86209772,
        #         184.36402194, 178.79969183, 173.26254351, 167.82877896,
        #         162.55175073, 157.48771028, 152.72463257, 148.37880019,
        #         144.54017745, 141.25386573, 138.47942179, 136.31249715,
        #         135.02114168, 134.9582344 , 135.94230509, 137.5893222 ,
        #         139.79939525, 142.56446464, 145.85405207, 149.65834817,
        #         154.32908838, 159.77824269]])
        
        # load initial conditions
        # init_cond_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20220215_ext_debugging_start.p'
        # with open(init_cond_file, 'rb') as f:
        #     mov_cline, target_img = pickle.load(f)
        
        
        x = 200; y = 200; rot = 185; stretch = 0.5
        EWCoeffs = [0,0,0,0,0]
        params = [x,y,rot,stretch, EWCoeffs[0],EWCoeffs[1],EWCoeffs[2],EWCoeffs[3],EWCoeffs[4]]
        worm_img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\scripts\20220217_extension_issue\test_files\bw1.png'
        target_img = np.zeros((400,400),dtype = 'uint8')
        worm_img = cv2.imread(worm_img_file,cv2.IMREAD_GRAYSCALE)
        shift_x = round((np.shape(target_img)[1]/2)-(np.shape(worm_img)[1]/2))
        shift_y = round((np.shape(target_img)[0]/2)-(np.shape(worm_img)[0]/2))
        target_img[shift_y:shift_y+np.shape(worm_img)[0],shift_x:shift_x+np.shape(worm_img)[1]] = worm_img
        
        deformable_model = Eigenworm_model()
        deformable_model.set_parameters(params)
        
        # fit the deformable model to the segmentation by gradient descent
        n_iter = 500
        lr = [20,20,20,.5,3,3,3,3,3]
        grad_step = [1,1,1,1,0.1,0.1,0.1,0.1,0.1]
        max_step = [2,2,2,0.02,0.1,0.1,0.1,0.1,0.1]
        save_dir = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\test_files\20220215_extension_debugging'
        show = True
        vid = False
        optimizer = Gradient_descent(deformable_model,
                                              target_img,
                                              n_iter, lr,
                                              grad_step, max_step,
                                              save_dir, show, vid)
        losses = optimizer.run()
        
        sfdafad

       
        
        # # initialize the model with pre-selected parameters
        # x = 200; y = 200; rot = -90; stretch = 1.0
        # EWCoeffs = [4,2,1,0,0]
        # parameters = [x,y,rot,stretch, EWCoeffs[0],EWCoeffs[1],EWCoeffs[2],EWCoeffs[3],EWCoeffs[4]]
        # model = Eigenworm_model()
        # model.set_parameters(parameters)
        # model.show_bw_image()
        
        # # demonstrate expansion and contraction in both directions
        # model.extend_in_track(direction = 1, amount = 0.1)
        # model.show_bw_image()
        # model.extend_in_track(direction = 1, amount = -0.2)
        # model.show_bw_image()
        # model.extend_in_track(direction = -1, amount = 0.1)
        # model.show_bw_image()
        # model.extend_in_track(direction = -1, amount = -0.1)
        # model.show_bw_image()
        # model.undo_last_extension()
        # model.show_bw_image()

        # # # demonstrate extend and then undo changes
        # orig_params = model.parameters
        # # print(orig_params)
        # mod.extend_in_track(direction = 1, amount = -0.2)
        # mod.show_bw_image()
        # # print(orig_params)
        # mod.undo_last_extension()        
        # mod.show_bw_image()
        
        # show the model and the target image
        
        # # gradient descent with out expansion
        # model = Eigenworm_model()
        # model.randomize_parameters()
        # # x = 150; y = 210; rot = 185; stretch = 1.0
        # # EWCoeffs = [0,0,0,0,0]
        # # parameters = [x,y,rot,stretch, EWCoeffs[0],EWCoeffs[1],EWCoeffs[2],EWCoeffs[3],EWCoeffs[4]]
        # # model = Eigenworm_model()
        # # model.set_parameters(parameters)
        
        # target_img = np.zeros((400,400),dtype = 'uint8')
        # worm_img = cv2.imread(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\bw5.png',cv2.IMREAD_GRAYSCALE)
        # shift_x = round((np.shape(target_img)[1]/2)-(np.shape(worm_img)[1]/2))
        # shift_y = round((np.shape(target_img)[0]/2)-(np.shape(worm_img)[0]/2))
        # target_img[shift_y:shift_y+np.shape(worm_img)[0],shift_x:shift_x+np.shape(worm_img)[1]] = worm_img
        
        # n_iter = 1000
        # lr = [20,20,20,.5,2,2,2,2,2]
        # #lr = [0,0,0,0,2,2,2,2,2]
        # grad_step = [1,1,1,1,0.1,0.1,0.1,0.1,0.1] # 0.1% of the range of that parameter
        # max_step = [2,2,2,0.02,0.1,0.1,0.1,0.1,0.1] # 1.0% of the range of that parameter
        # save_dir = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files'
        # show = True
        # vid = True
        
        
        # optimizer = Gradient_descent(model, target_img, n_iter, lr, grad_step, max_step, save_dir, show, vid)
        # optimizer.run()
        

        #optimizer.run()
        #new_centerline = optimizer.
        
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    


