# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:56:56 2021



# issues and improvements:
    -centerlines are being saved as integers
    -automatically populate parameters based on scale and sample frame
    -load the parameters from another video
    -choose separate parameters for each video, or 'apply all'
    -change autoset tile_width and overlap according to scale or set manually

@author: PDMcClanahan
"""
import numpy as np
import os
import cv2
import copy
import csv
import tkinter as tk
from scipy import interpolate
from scipy.spatial import distance
from tkinter import simpledialog

# for find_centerline
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d

# for mask RCNN tracking
import sys
sys.path.append(os.path.split(__file__)[0])
import mrcnn_module as mrcnn
import deformable_worm_module as def_worm


class Tracker:
    
    num_vids = 0
    default_centerline_method = 'ridgeline'
    default_tracking_method = 'intensity'
    default_tracking_method = 'mask_RCNN'
    
    tshooting_outputs = [1]
    
    metaparameters = {
        'intensity_sigma_area' : 3, # = pi*(sigma*(1/um_per_pix))^2
        'intensity_area_bounds' : (200,400), # in um^2
        'mRCNN_sigma_area' : 3,
        'mRCNN_area_bound' : (200,400),
        'centerline_npts' : 50,
        'edge_proximity_cutoff' : 10,
        }
    
    size_factors = {
        'dauer' : 1.0,
        'IJ' : 1.2,
        'L3' : 1.0,
        'L4' : 1.2,
        'young adult' : 2.0,
        'adult' : 2.5,
        }
        

    def __init__(self, vid_file, um_per_pix = None, worm_type = 'dauer'):
        self.vid_path, self.vid_name = os.path.split(vid_file) 
        self.save_path = self.vid_path + '//' + self.vid_name[:-4] + '_tracking'
        self.save_path_troubleshooting = self.save_path + '//troubleshooting'
        
        self.tracking_method = 'mask_RCNN'
        #self.tracking_method = 'intensity'
        self.centerline_method = 'ridgeline'
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(self.save_path_troubleshooting):
            os.mkdir(self.save_path_troubleshooting)
        
        if os.path.isfile(self.save_path+'\\tracking_parameters.csv'):
            print('Re-loading parameters')
            try:
                self.load_parameter_csv()
            except:
                import pdb
                import sys
                import traceback
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
        else:
            #self.auto_set_parameters(worm_type)
            self.parameters = {
            'human_checked' : False,
            'bkgnd_meth' : 'max_merge',
            'bkgnd_nframes' : 10,
            'k_sig' : 1.5,
            'bw_thr' : 10,
            'area_bnds' : (600,1500),
            'd_thr' : 10,
            'del_sz_thr' : '',
            'um_per_pix' : '',
            'min_f' : 300
            }
            #self.save_params_csv('tracking_parameters')
        
        self.model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Steinernema\20220127_full_frame_Sc_on_udirt_4.pt'
        self.vid = cv2.VideoCapture(self.vid_path+'//'+self.vid_name)
        self.num_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_background()
        self.dimensions = (np.shape(self.background))
        
        # load centroids, centerlines, parameters, etc from prior tracking
        
        
        Tracker.num_vids += 1
    
    
    def auto_set_parameters(self, worm_type):
        factor = Tracker.size_factors[worm_type]
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.parameters = {
            'human_checked' : False,
            'bkgnd_meth' : 'max_merge',
            'bkgnd_nframes' : 10,
            'k_sig' : 1.5,
            'bw_thr' : 10,
            'area_bnds' : (600,1500),
            'd_thr' : 10,
            'del_sz_thr' : '',
            'um_per_pix' : '',
            'min_f' : 50
            }
        
        
    
    def set_parameters(self,human_checked,bkgnd_meth,bkgnd_nframes,k_sig,
                       bw_thr,area_bnds,d_thr,del_sz_thr,um_per_pix,min_f):
        
        self.parameters['human_checked'] = human_checked
        self.parameters['bkgnd_meth'] = bkgnd_meth
        self.parameters['bkgnd_nframes'] = bkgnd_nframes
        self.parameters['k_sig'] = k_sig
        self.parameters['bw_thr'] = bw_thr
        self.parameters['area_bnds'] = area_bnds
        self.parameters['d_thr'] = d_thr
        self.parameters['del_sz_thr'] = bkgnd_meth
        self.parameters['um_per_pix'] = um_per_pix
        self.parameters['min_f'] = min_f
        
    
    
    
    @classmethod
    def set_centerline_method():
        pass
    
    # creates images for display in parameter GUI
    def show_segmentation(self, f=0):
        
        print(self.tracking_method)
        
        # set up mask RCNN if needed
        if self.tracking_method == 'mask_RCNN' and 'self.model_file' not in locals():
            # self.tile_width = 500 # set equal to the size of the training images (assumed to be square)
            # self.overlap = 250 # set equal to max dimension of an object to be segmented
            self.model, self.device = mrcnn.prepare_model(self.model_file)
            self.scale_factor = 0.5
            self.param_gui_f = -1 # tracks for which frame 'diff' was calculated
        
        # read in frame f
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # get parameters (makes code below more readable)
        k_size = (round(self.parameters['k_sig']*3)*2+1,round(self.parameters['k_sig']*3)*2+1)
        k_sig = self.parameters['k_sig']
        bw_thr = self.parameters['bw_thr']
        area_bnds = self.parameters['area_bnds']
        
        # make bw image
        if self.tracking_method == 'intensity':
            self.diff = (np.abs(img.astype('int16') - self.background.astype('int16'))).astype('uint8')
        elif self.tracking_method == 'mask_RCNN':
            if self.param_gui_f == f: # do not recalculate if not necessary
                print('hi')    
                pass
            else:
                self.diff = mrcnn.segment_full_frame(img, self.model, self.device, self.scale_factor)
                self.param_gui_f = f
        smooth = cv2.GaussianBlur(self.diff,k_size,k_sig,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)    
        
        # find all connected components
        cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
        cc_map = np.uint8(cc[1]); 
        cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)

        # eliminate objects that are too big or too small, make a BW image
        # without eliminated objects, and find centerlines of included
        # objects, flag bad centerlines
        bw_ws = np.zeros(np.shape(bw),dtype = 'uint8')
        centroids = list()
        clines_f = list()
        cline_flags = list()
        for cc_i in cc_is:
            cc_sz = cc[2][cc_i][4]
            if cc_sz > area_bnds[0] and cc_sz < area_bnds[1]:
                bw_ws[np.where(cc_map==cc_i)] = 255
                centroids.append(copy.deepcopy(cc[3][cc_i]))
                bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                bw_w[np.where(bw_w == cc_i)]=255
                bw_w[np.where(bw_w!=255)]=0
                cline = np.uint16(np.round(Tracker.find_centerline(bw_w)[0]))
                cline_flags.append(Tracker.flag_bad_centerline(cline))
                cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1] 
                clines_f.append(copy.copy(cline))
                
        
        # setup overlay text based on image size
        font_factor = self.dimensions[1]/720
        f_face = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = .5 * font_factor
        f_thickness = round(2 * font_factor)
        f_color = (0,0,0)
        linewidth = round(1 * font_factor)
        offset = round(50 * font_factor)
        
        # create 'final' image showing identified worms
        final_HSV = cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HSV)  
        
        # add red shading to all bw blobs
        final_HSV[:,:,0][np.where(bw==255)] = 120 # set hue (color)
        final_HSV[:,:,1][np.where(bw==255)] = 80 # set saturation (amount of color, 0 is grayscale)
    
        # change shading to green for those that are within the size bounds
        final_HSV[:,:,0][np.where(bw_ws==255)] = 65 # set hue (color)
        
        # convert image to BGR (the cv2 standard)
        final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
        
        # could also outline blobs within size bounds
        # final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
        # contours, hierarchy = cv2.findContours(bw_ws, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(final, [contours[0]], 0, (0, 255, 0), 1) #drawing contours
        
        # label blobs detected as worms with centerline
        for track in range(np.shape(centroids)[0]):
            # cline
            pts = np.int32(clines_f[track])
            pts = pts.reshape((-1,1,2))
            final = cv2.polylines(final, pts, True, (0,255,0), linewidth)
        
        # label the size of all blobs
        for cc_i in cc_is[1:]:
            cc_sz = cc[2][cc_i][4]
            text = str(cc_sz)
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            text_pos = copy.copy(cc[3][cc_i]) # deepcopy avoids changing objs below
            text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
            text_pos[1] = text_pos[1] + 30
            text_pos = tuple(np.uint16(text_pos))
            if cc_sz > area_bnds[0] and cc_sz < area_bnds[1]:
                final = cv2.putText(final,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
            else:
                final = cv2.putText(final,text,text_pos,f_face,f_scale,(50,50,50),f_thickness,cv2.LINE_AA)
     
        # show the distance threshold
        if self.parameters['d_thr'] is not None:
            d_thr = self.parameters['d_thr']
            text = 'd='+str(d_thr)
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            pt1 = [np.shape(img)[1]-offset,np.shape(img)[0]-offset]
            pt2 = [pt1[0]-d_thr,pt1[1]]
            text_pos = np.array((((pt1[0]+pt2[0])/2,pt1[1])),dtype='uint16')
            text_pos[0] = text_pos[0] - text_size[0]/2 # x centering 
            text_pos[1] = text_pos[1] - 5 # y offset   
            final = cv2.polylines(final, np.array([[pt1,pt2]]), True, (0,0,255), linewidth)
            final = cv2.putText(final,text,tuple(text_pos),f_face,f_scale,(0,0,255),f_thickness,cv2.LINE_AA)
            del pt1, pt2
        
        return img, self.diff, smooth, bw, bw_ws, final
        
        
    def track(self,fix_centerlines = True):
        
        
        # set up model if using mask RCNN
        if self.tracking_method == 'mask_RCNN':
            # self.tile_width = 500 # set equal to the size of the training images (assumed to be square)
            # self.overlap = 250 # set equal to max dimension of an object to be segmented
            self.scale_factor = 0.5 # scale factor of the frames used to train the mRCNN
            self.model, self.device = mrcnn.prepare_model(self.model_file)
        
        
        # set up loop, vars to hold centroid
        self.centroids_raw = []
        self.first_frames_raw = []
        if self.centerline_method  != 'none':
            self.centerlines_raw = []
            self.centerline_flags_raw = []
            self.angles_end_1_raw = []
            self.angles_end_2_raw = []
        else:
            self.centerlines_raw = None
            self.centerline_flags_raw = None
            self.angles_end_1_raw = None
            self.angles_end_2_raw = None
        
        
        # tracking loop
        indices = np.linspace(0,self.num_frames-1,int(self.num_frames)); i = 0;
        for i in indices:
            print('Finding worms in frame '+str(int(i+1))+' of '+str(int(self.num_frames)))
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            centroids_frame, centerlines_frame, centerline_flags_frame, \
                    angles_end_1_frame, angles_end_2_frame = \
                    self.find_worms(img)


            
            self.stitch_centroids(centroids_frame, centerlines_frame,
                                  centerline_flags_frame, angles_end_1_frame,
                                  angles_end_2_frame, i)
            

        # cleanup
        self.remove_short_tracks()
        if fix_centerlines:
            self.fix_centerlines()
        
        self.align_centerlines()
        #self.orient_head_tail()
        
        # save results
        self.save_centroids(self.centroids, self.first_frames, self.save_path,
                            'centroids')
        self.save_centerlines(self.centerlines, self.centerline_flags,
                              self.first_frames, self.save_path)
        self.save_end_angles(self.angles_end_1, self.save_path, '1')
        self.save_end_angles(self.angles_end_2, self.save_path, '2')
        
        # make tracking video
        out_scale = 0.5
        self.create_summary_video(out_scale)
        
        

    def save_params_csv(self,save_name = 'tracking_parameters'):
        save_path = self.vid_path + '\\' + os.path.splitext(self.vid_name)[0] + '_tracking'
        if save_name == 'mRCNN_tracking_parameters':
            params = self.parameters
        else:
            params = self.parameters
        
        if not os.path.exists(save_path):
            print('Creating directory for tracking parameters and output: '+save_path)
            os.makedirs(save_path)
        
        save_file_csv = save_path + '\\' + save_name + '.csv'
        
        with open(save_file_csv, mode='w',newline="") as csv_file: 
            keys = list(params.keys())
            
            parameters_writer = csv.writer(csv_file, delimiter=';',
                                           quotechar='"',
                                           quoting=csv.QUOTE_MINIMAL)
            
            row = ['Parameter','Value']
            parameters_writer.writerow(row)
            
            for r in range(len(params)):
                row = [keys[r],str(params[keys[r]])]
                parameters_writer.writerow(row)
            
        print("Tracking parameters saved in " + save_file_csv )

    

    def load_parameter_csv(self):
        self.parameters = dict()
        csv_filename = self.save_path+'\\tracking_parameters.csv'
        
        with open(csv_filename, newline="") as csv_file: 
            parameters_reader = csv.reader(csv_file, delimiter=';', quotechar='"')
            for r in parameters_reader:
                if r[0] == 'Parameter' or r[1] == '':
                    pass
                elif r[0] == 'human_checked':
                    if r[1]  == 'True' or r[1]  == 'TRUE':
                        r[1] = True
                    else:
                        r[1] = False
                    
                elif r[0] == 'area_bnds':
                    transdict = {91: None, 93: None, 40: None, 41: None, 44: None}
                    r[1] = r[1].translate(transdict).split(sep=' ')
                    r[1] =  [int(r[1][n]) for n in range(len(r[1]))]
                elif r[0] == 'k_sig' or r[0] == 'um_per_pix':
                    try:
                        r[1] = float(r[1])
                    except:
                        pass
                    
                elif r[0] == 'bkgnd_meth': # string
                    pass
                else: # all other params should be integers
                    r[1] = int(r[1])
                 
                if r[0] != 'Parameter':
                    self.parameters[r[0]] = r[1]
        

    def get_background(self):
        print('Calculating background image...')
        
        inds = np.round(np.linspace(0,self.num_frames-1,self.parameters['bkgnd_nframes'])).astype(int)
             
        for i in inds:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            if i == inds[0]:
                ret,stack = self.vid.read(); stack = cv2.cvtColor(stack, cv2.COLOR_BGR2GRAY)
                stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1)) 
            else:
                ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.reshape(img,(img.shape[0],img.shape[1],1)) 
                stack = np.concatenate((stack, img), axis=2)
                stack = np.amax(stack,2)
                stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1)) 
        
        stack = np.squeeze(stack)
        self.background = stack  
        
        
    
    def find_worms(self,img):
        
        k_size = (round(self.parameters['k_sig']*3)*2+1,round(self.parameters['k_sig']*3)*2+1)
        k_sig = self.parameters['k_sig']
        bw_thr = self.parameters['bw_thr']
        area_bnds = self.parameters['area_bnds']
        
        if self.tracking_method == 'intensity':
            diff = (np.abs(img.astype('int16') - self.background.astype('int16'))).astype('uint8')
        elif self.tracking_method == 'mask_RCNN':
            #diff = mrcnn.segment_frame_by_tiling(img, self.model, self.device, self.tile_width, self.overlap)
            diff = mrcnn.segment_full_frame(img, self.model, self.device, self.scale_factor)
        
        smooth = cv2.GaussianBlur(diff,k_size,k_sig,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
        
        # cc: # objs, labels, stats, centroids
        #  -> stats: left, top, width, height, area
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

                if np.min(obj_inds_r) <= self.metaparameters['edge_proximity_cutoff'] or np.min(obj_inds_c) <= self.metaparameters['edge_proximity_cutoff']:
                    hits_edge = True
                elif np.max(obj_inds_r) >= np.shape(cc[1])[0]-1-self.metaparameters['edge_proximity_cutoff'] or np.max(obj_inds_c) >= np.shape(cc[1])[1]-1-self.metaparameters['edge_proximity_cutoff']:
                    hits_edge = True
                    
                if hits_edge is False:
                    centroids.append(copy.deepcopy(cc[3][cc_i]))
                    
                    # find the centerline
                    if self.centerline_method != 'none':
                        bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                        bw_w[np.where(bw_w == cc_i)]=255
                        bw_w[np.where(bw_w!=255)]=0
                        centerline, angle_end_1, angle_end_2 = Tracker.find_centerline(bw_w,self.centerline_method)
                        centerline = np.uint16(np.round(centerline))
                        centerline_flag = Tracker.flag_bad_centerline(centerline)
                        centerline[:,0] += cc[2][cc_i][0]
                        centerline[:,1] += cc[2][cc_i][1]
                        centerline = centerline[np.newaxis,...]
                        centerlines.append(copy.copy(centerline))
                        centerline_flags.append(centerline_flag)
                        
                        angles_end_1.append(angle_end_1)
                        angles_end_2.append(angle_end_2)
                        
        # re-arrange centroids
        
        if self.centerline_method != 'none':
            return centroids, centerlines, centerline_flags, angles_end_1, angles_end_2
        else:
            return centroids
    
    

    
    def stitch_centroids(self, centroids_frame, centerlines_frame, 
                         centerline_flags_frame, angles_end_1_frame, 
                         angles_end_2_frame, f):
        
        # make a list of the indices of objects tracked in the previous frame
        # as well as their centroids
        prev_obj_inds = []
        centroids_prev = []
        j = 0
        for i in range(len(self.centroids_raw)):
            if self.first_frames_raw[i] + len(self.centroids_raw[i]) == f:
                prev_obj_inds.append(i)
                centroids_prev.append([self.centroids_raw[i][-1]])
                j += 1
                
        # if no objects were tracked in the previous frame, all objects in the
        # current frame are new objects and no stitching is needed
        if len(centroids_prev) == 0:
            for i in range(len(centroids_frame)):
                self.centroids_raw.append([centroids_frame[i]])
                self.first_frames_raw.append(int(f))
                if self.centerline_method != 'none':
                    self.centerlines_raw.append([centerlines_frame[i]])
                    self.centerline_flags_raw.append([centerline_flags_frame[i]])
                    self.angles_end_1_raw.append([angles_end_1_frame[i]])
                    self.angles_end_2_raw.append([angles_end_2_frame[i]])
        
        # if no objects were tracked in the current frame, do nothing
        elif len(centroids_frame) == 0:
            pass
        
        # otherwise create a matrix of the distances between the object 
        # centroids tracked in the previous frame and those tracked in the
        # current frame
        else:
            d_mat = np.empty((len(centroids_prev),len(centroids_frame)))
            for row, cent_prev in enumerate(centroids_prev):            
                for col, cent_frame in enumerate(centroids_frame):
                    d_mat[row,col] = np.linalg.norm(cent_prev-cent_frame)    
            
            # find the closest centroids, second closest, etc., crossing off
            # matched objects until either all objects from the previous frame
            # are matched, or none of them are within the max distance of per
            # frame travel
            num_to_pair =  np.min(np.shape(d_mat))
            search = True
            pair_list = list()
    
            #if np.shape(d_mat)[0]>0 and np.shape(d_mat)[1]>0:
            while search:
                min_dist = np.nanmin(d_mat)
                if min_dist < self.parameters['d_thr']:
                    result = np.where(d_mat == np.nanmin(d_mat))  
                    pair_list.append((result[0][0],result[1][0]))
                    d_mat[result[0][0],:]=np.nan
                    d_mat[:,result[1][0]]=np.nan
                else:
                    search = False
                
                if len(pair_list) == num_to_pair:
                    search = False
            
            # The tracks of objects tracked in the last frame but not matched
            # in this frame are dropped (nothing more appended to their entry
            # in the centroids_raw list), objects in the current frame that
            # were matched to objects in the previous frame are appended to
            # the corresponding item in centorids_raw, and new objects
            # detected in this frame are added as new items in the
            # centroids_raw list. If applicable, the same is done for
            # centerlines and end angles.

            # objects tracked from the previous frame
            tracked_inds = []
            for i in range(len(pair_list)):
                tracked_inds.append(pair_list[i][1])
                self.centroids_raw[prev_obj_inds[pair_list[i][0]]].append(centroids_frame[pair_list[i][1]])
                if self.centerline_method != 'none':
                    self.centerlines_raw[prev_obj_inds[pair_list[i][0]]].append(centerlines_frame[pair_list[i][1]])
                    self.centerline_flags_raw[prev_obj_inds[pair_list[i][0]]].append(centerline_flags_frame[pair_list[i][1]])
                    self.angles_end_1_raw[prev_obj_inds[pair_list[i][0]]].append(angles_end_1_frame[pair_list[i][1]])
                    self.angles_end_2_raw[prev_obj_inds[pair_list[i][0]]].append(angles_end_2_frame[pair_list[i][1]])
            
            # newly tracked objects
            for i in range(len(centroids_frame)):
                if i not in tracked_inds:
                    self.centroids_raw.append([centroids_frame[i]])
                    self.first_frames_raw.append(int(f))
                    if self.centerline_method != 'none':
                        self.centerlines_raw.append([centerlines_frame[i]])
                        self.centerline_flags_raw.append([centerline_flags_frame[i]])
                        self.angles_end_1_raw.append([angles_end_1_frame[i]])
                        self.angles_end_2_raw.append([angles_end_2_frame[i]])

          
    def remove_short_tracks(self):
        
        print('Eliminating worms tracked for fewer than '+str(self.parameters['min_f'])+' frames...')
        
        self.centroids = copy.deepcopy(self.centroids_raw)
        self.first_frames = copy.deepcopy(self.first_frames_raw)
        self.centerlines = copy.deepcopy(self.centerlines_raw)
        self.centerline_flags = copy.deepcopy(self.centerline_flags_raw)
        self.angles_end_1 = copy.deepcopy(self.angles_end_1_raw)
        self.angles_end_2 = copy.deepcopy(self.angles_end_2_raw)
        
        for w in reversed(range(len(self.centroids))):
            if len(self.centroids[w]) < self.parameters['min_f']:
                print('Eliminating worm track '+str(w))
                self.centroids.pop(w)
                self.first_frames.pop(w)
                if self.centerlines is not None:
                    self.centerlines.pop(w)
                    self.centerline_flags.pop(w)
                    self.angles_end_1.pop(w)
                    self.angles_end_2.pop(w)

        print('Done removing short traces!')


    def fix_centerlines(self):
        '''Fixes flagged centerlines by fitting a deformable model based on
        the nearest non-flagged or fixed centerline to the bw image that
        resulted in the flagged centerline'''
        
        
        self.centerlines_unfixed = copy.deepcopy(self.centerlines)
        self.centerline_flags_unfixed = copy.deepcopy(self.centerline_flags)
        fw = 200 # frame_width, should be changed to depend on may worm size
        k_size = (round(self.parameters['k_sig']*3)*2+1,round(self.parameters['k_sig']*3)*2+1)
        from scipy.ndimage.morphology import distance_transform_edt as dist_trans
        # from skimage.segmentation import flood_fill # importing this causes crash
        
        flags_new = copy.deepcopy(self.centerline_flags)
        

        # determine the number of centerlines that need to be fixed, as this
        # is a time-consuming step
        num_flagged = 0
        num_fixed = 0
        for i in range(len(self.centerline_flags)): 
            num_flagged += len(np.where(self.centerline_flags[i])[0] == 1)
        
        # fix centerlines worm by worm
        for w in range(len(self.centerlines)):
            flags = np.array(self.centerline_flags[w]) # new flags after fixing
            if len(np.where(flags==1)[0]) == len(flags):
                print('Worm ' + str(w) + ' has no non-flagged centerlines')
            else:
                flags_prog = copy.copy(flags) # copy for distance transform purposes
                
                # fix centerlines in order of closeness to a good centerline
                while len(np.where(flags_prog==1)[0]) > 0:
                    # if w == 13:
                    print('Fixing flagged centerline '+ str(num_fixed+1) + ' of ' + str(num_flagged))
                    
                    # choose a centerline to correct based on proximity to a non-
                    # flagged or fixed centerline and retrieve the nearest non-flagged
                    # centerline
                    
                    dists = dist_trans(flags_prog)
                    f = np.where(dists == 1)[0][0]
                    
                    if f > 0 and flags_prog[f-1] == 0: # frame before is OK
                        moving_centerline = self.centerlines[w][f-1]
                    elif f < len(self.centerlines[w])-1 and flags_prog[f+1] == 0: # frame after is OK
                        moving_centerline = self.centerlines[w][f+1]
                    midpoint = round(len(moving_centerline[0])/2)
                    
                    # retrieve the target image (bw segmentation of the worm that
                    # caused the flagged centerline)
                    unshifted_f = self.first_frames[w] + f
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, unshifted_f)
                    ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if self.tracking_method == 'intensity':
                        diff = (np.abs(img.astype('int16') - self.background.astype('int16'))).astype('uint8')
                    elif self.tracking_method == 'mask_RCNN':
                        diff = mrcnn.segment_full_frame(img, self.model,
                                                        self.device,
                                                        self.scale_factor)
                    smooth = cv2.GaussianBlur(diff,k_size,
                                              self.parameters['k_sig'],
                                              cv2.BORDER_REPLICATE)
                    thresh,bw = cv2.threshold(smooth,self.parameters['bw_thr'],255,cv2.THRESH_BINARY)
                    height, width = bw.shape[:2]
                    mask = np.zeros((height + 2, width + 2), np.uint8)
                    cv2.floodFill(bw, mask, (moving_centerline[0][midpoint][0].astype(np.int16),moving_centerline[0][midpoint][1].astype(np.int16)), 127)
                    bw[np.where(bw==255)]=0
                    bw[np.where(bw==127)]=255
                    # add a frame
                    bw_framed = np.zeros((np.shape(bw)[0]+2*fw,
                                          np.shape(bw)[1]+2*fw),
                                         dtype = 'uint8')
                    bw_framed[fw:-fw,fw:-fw] = bw
                    target_coords = moving_centerline[0][midpoint].astype(np.int16)+200
                    target_image = bw_framed[(target_coords[1]-fw):(target_coords[1]+fw),(target_coords[0]-fw):(target_coords[0]+fw)]
                    moving_centerline_shifted = copy.copy(moving_centerline)
                    moving_centerline_shifted[0,:,0] = moving_centerline[0,:,0] - moving_centerline[0][midpoint][0] + 200
                    moving_centerline_shifted[0,:,1] = moving_centerline[0,:,1] - moving_centerline[0][midpoint][1] + 200
                    moving_centerline_shifted = np.swapaxes(np.squeeze(moving_centerline_shifted,0),1,0)
    
                    # import matplotlib.pyplot as plt
                    # plt.imshow(target_image, cmap = 'gray')
                    # plt.plot(moving_centerline_shifted[0,:],moving_centerline_shifted[1,:],'r--')
                    # plt.show()
    
                    # initialize a deformable model
                    deformable_model = def_worm.Eigenworm_model()
                    deformable_model.set_centerline(moving_centerline_shifted)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(deformable_model.bw_image, cmap = 'gray'); plt.show()
                    
                    # fit the deformable model to the segmentation by gradient descent
                    n_iter = 100
                    lr = [20,20,20,.5,3,3,3,3,3]
                    grad_step = [1,1,1,1,0.1,0.1,0.1,0.1,0.1]
                    max_step = [2,2,2,0.02,0.1,0.1,0.1,0.1,0.1]
                    save_dir = self.save_path_troubleshooting
                    show = False
                    vid = False
                    optimizer = def_worm.Gradient_descent(deformable_model,
                                                          target_image,
                                                          n_iter, lr,
                                                          grad_step, max_step,
                                                          save_dir, show, vid)
                    optimizer.run()
                    # plt.imshow(deformable_model.bw_image, cmap = 'gray'); plt.show()
    
                    # check the new centerline
                    
                    # save the new centerline and change the flag value to 2
                    new_centerline = deformable_model.centerline
                    new_centerline = new_centerline.swapaxes(0,1)
                    new_centerline = np.expand_dims(new_centerline,0)
                    new_centerline[0,:,0] += moving_centerline[0][midpoint].astype(np.int16)[0]-200
                    new_centerline[0,:,1] += moving_centerline[0][midpoint].astype(np.int16)[1]-200
                    self.centerlines[w][f] = new_centerline
                    flags[f] = 2
                    flags_prog[f] = 0
                    
                    # check the new centerline and change the flag to 2 (fixed) if it passes
                    num_fixed +=1
                    
                self.centerline_flags[w] = copy.deepcopy(flags.tolist())


    def align_centerlines(self):    
        
        
        def dist(p1,p2):
            d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
            return d
        
        
        for w, clines in enumerate(self.centerlines):
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
                    temp_1 = copy.copy(self.angles_end_1[w][f+1])
                    temp_2 = copy.copy(self.angles_end_2[w][f+1])
                    self.angles_end_1[w][f+1] = temp_2
                    self.angles_end_2[w][f+1] = temp_1
            
        

    def orient_head_tail(self):
        # import pdb; pdb.set_trace()
        
        
            
        # look up which end is sharper (see Wang and Wang 2013 PLOS and
        # Leifer et al 2011 Nat Methods)
        
        # compute direction of movement (e.g. Roussel et al 2007)
        
        # neural networm (Mane et al)

        # find the total movement of each end
        # COULD USE MANY POINTS (see pg 14 of E. Yemini's dissertation)
        ends_mov = [0,0]
        for f in range(len(clines)-1):
            ends_mov[0] += dist(clines[f][0][0].astype(np.float32),
                             clines[f+1][0][0].astype(np.float32))
            ends_mov[-1] += dist(clines[f][0][-1].astype(np.float32),
                             clines[f+1][0][-1].astype(np.float32))
            
        # determine which end is the head and flip if necessary
        if ends_mov[-1] > ends_mov[0]:
            clines = np.flip(clines,-2)
            
            
        # find the intensity profile by sampling along the centerline
        shape = np.shape(clines)
        i_profile = np.empty((shape[2],shape[0]))
        frames = np.linspace(self.first_frames[w],self.first_frames[w])
        for f, cline in enumerate(clines):
            ret = self.vid.set(cv2.CAP_PROP_POS_FRAMES, frames[f])
            ret, img = self.vid.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            i = cline[0,:,0].astype(np.uint16)
            j = cline[0,:,1].astype(np.uint16)
            i_profile[:,f] = img[j,i]
        filename = self.save_path_troubleshooting + '//worm_'+str(w) + \
            '_intensity.bmp'
        try:
            ret = cv2.imwrite(filename, i_profile)
        except:
            pass
        

    @staticmethod
    def save_centroids(centroids, first_frames, save_path, 
                       save_name = 'centroids'):
        
        if not os.path.exists(save_path):
            print('Creating directory for centroids csv and other output: '
                  +save_path)
            os.makedirs(save_path)
        
        save_file_csv = save_path + '\\' + save_name + '.csv'
        
        with open(save_file_csv, mode='w',newline="") as csv_file: 
            
            writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            row = ['First Frame of Track',
                   'X and then Y Coordinates on Alternating Rows']
            writer.writerow(row)
            
            for t in range(len(centroids)):
                x_row = [str(first_frames[t])]
                y_row = ['']
                for i in np.arange(0,len(centroids[t])):
                    x_row.append(str(round(float(centroids[t][i][0]),1)))
                    y_row.append(str(round(float(centroids[t][i][1]),1)))
                writer.writerow(x_row)
                writer.writerow(y_row)
            
        print("Centroids saved as " + save_file_csv )
    
    
    
    @staticmethod
    def load_centroids(save_path, save_name = 'centroids'):
        centroids = []
        return centroids
    
    @staticmethod
    def save_centerlines(centerlines, centerline_flags, first_frames, 
                         save_path):

        if not os.path.exists(save_path + '\\centerlines'):
            print('Creating directory for centerlines csvs and other output: '
                  +save_path+'\\centerlines')
            os.makedirs(save_path+'\\centerlines')
        
        for w in range(len(centerlines)):
            save_file_csv = save_path + '\\centerlines\\' + \
                'centerlines_worm_' + "{:06d}".format(w) + '.csv'
            
            with open(save_file_csv, mode='w',newline="") as csv_file: 
                
                writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                
                # write row 1: frame number
                row = ['frame']
                for f in np.arange(first_frames[w],
                                   first_frames[w]+len(centerlines[w])):
                    row.append(str(int(f+1)))
                writer.writerow(row)
                
                # write row 2: centerline flag
                row = ['flag']
                for f in np.arange(len(centerline_flags[w])):
                    row.append(str(int(centerline_flags[w][f])))
                writer.writerow(row)
                
                # write remaining rows: centerline point coordinates
                for xy in range(2):
                    for p in range(np.shape(centerlines[w])[2]):
                        if xy == 0:
                            row = ['x'+str(p)]
                        else:
                            row = ['y'+str(p)]
                        for t in range(len(centerlines[w])):
                            row.append(str(round(float(
                                centerlines[w][t][0,p,xy]),1))
                                )
                        writer.writerow(row)
                    
                
        print("Centerlines saved in " + save_path + '\\centerlines')
        
    @staticmethod
    def save_end_angles(end_angles, save_path, number):
        '''Saves values in <end_angles> in a .csv in <save_path>. <number>,
        referring to the end of the worm to which the angles pertain, is used
        in the filename'''
        
        if not os.path.exists(save_path):
            print('Creating directory for end angles and other output: ' +
                  save_path)
            os.makedirs(save_path)

        save_file_csv = save_path + '\\' + 'end_' + number + '_angles.csv'
            
        with open(save_file_csv, mode='w',newline="") as csv_file: 
                
                writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                
                # write row 1: headings
                row = ['worm','angles']
                writer.writerow(row)
                
                # write remaining rows: worm numbers and angles
                for w, angs in enumerate(end_angles):
                    row = [str(w)]
                    for ang in angs:
                        row.append(str(ang))
                    writer.writerow(row)
                
        print("End " + str(number) + " angles saved in " + save_path)
    
    def load_centerlines(self):
        self.centerlines = list()
        self.centerline_flags = list()
        centerline_files = os.listdir(self.save_path + '\\centerlines')
        w = 0
        for file in range(len(centerline_files)):
            csv_filename = self.save_path + '\\centerlines\\' + \
                           centerline_files[file]
            if csv_filename[-9:] == 'worm'+str(w)+'.csv':
                with open(csv_filename, newline="") as csv_file: 
                    centerlines_reader = csv.reader(csv_file, delimiter=',',
                                                    quotechar='"')
                    for r in centerlines_reader:
                        if r[0]=='frame':
                            numf = 1+int(r[-1])-int(r[1]); f = 0
                            numpts = self.metaparameters['centerline_npts']
                            centerlines_worm = np.empty((numf,1,numpts,2))
                        elif r[0] == 'x0' or r[0] == 'y0':
                            p = 0
                            
                        if r[0] == 'flag':
                            centerline_flags_worm = list()
                            for ff in range(len(r)-1):
                                centerline_flags_worm.append(int(r[ff+1]))
                        
                        if r[0][0] == 'x':
                            for f in range(len(r)-1):
                                centerlines_worm[f,0,p,0] = float(r[f+1])
                            p+=1
                        elif r[0][0] == 'y':
                            for f in range(len(r)-1):
                                centerlines_worm[f,0,p,1] = float(r[f+1])
                            p+=1        
            w += 1
            self.centerlines.append(list(centerlines_worm))
            self.centerline_flags.append(centerline_flags_worm)

                        
    
    
    @staticmethod
    def draw_scale(img):
        
        imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        clicks_x, clicks_y = [],[]
        z = []
        um_per_pix = np.nan
        
        def draw_line(event,x,y,flags,param):
            nonlocal imgc
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(clicks_x) < 2:
                    clicks_x.append(x)
                    clicks_y.append(y)
                    
                if len(clicks_x) == 1:
                    cv2.circle(imgc,(x,y),1,(0,0,255),-1)
                elif len(clicks_x) == 2:
                    cv2.line(imgc, (clicks_x[-2],clicks_y[-2]),
                             (clicks_x[-1],clicks_y[-1]), (0,0,255), 2)
                    cv2.circle(imgc,(clicks_x[-1],clicks_y[-1]),2,(0,0,255),
                               -1)
                    z.append(1)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(clicks_x) >0:
                    clicks_x.pop()
                    clicks_y.pop()
                
                imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                if len(clicks_x) ==0:
                    pass
                if len(clicks_x) ==1:
                    cv2.circle(imgc,(clicks_x[0],clicks_y[0]),2,(0,0,255),-1)
                if len(clicks_x) > 1:
                    print('x_clicks unexpectedly long')
                 
        cv2.namedWindow('Scale image',cv2.WINDOW_NORMAL)
        cv2.imshow('Scale image',imgc)
        cv2.setMouseCallback('Scale image',draw_line)
        
        while(1):
            cv2.imshow('Scale image',imgc)
            k = cv2.waitKey(20) & 0xFF
            if len(z)>0:
                print('done')
                cv2.imshow('Scale image',imgc) # display pt 2
                k = cv2.waitKey(20) & 0xFF
                
                dialogue_box = tk.Tk()
                dialogue_box.withdraw() # prevents the annoying extra window
                um = simpledialog.askfloat("Input",
                                           "Enter the distance in microns:",
                                           parent=dialogue_box,
                                           minvalue=0, maxvalue=1000000)
                
                dialogue_box.destroy()
                #dialogue_box.quit()
                #dialogue_box.mainloop()
                    
                if um is not None:
                    z.append(1)
                    cv2.destroyAllWindows()
                    break
                else:
                    z = []
        
        # get user input for scale
        pix = np.sqrt((clicks_x[-1]-clicks_x[0])**2+(clicks_y[-1]-clicks_y[0])**2)
        um_per_pix = um / pix
        
        scale_img = imgc
        
        return um_per_pix, scale_img
        
    ######################
    
    @staticmethod
    def find_centerline(bw, method = 'ridgeline', debug = False):
        '''Takes a binary image of a worm and returns the centerline. The ends
        are detected by finding two distant minima of the smoothed interior 
        angle. Centerline points are detected by finding points that are local
        minima of the distance transform in both the x and y direction. These
        points are resampled and returned along with the smoothed interior 
        angles at each end.'''
        
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
            #dangles_unwr = np.unwrap(dangles)
            dangles_unwr = copy.copy(dangles)
            for a in range(len(dangles_unwr)):
                if dangles_unwr[a] > np.pi:
                    dangles_unwr[a]  = -(2*np.pi-dangles_unwr[a] )
                elif dangles_unwr[a] < -np.pi:
                    dangles_unwr[a] = 2*np.pi+dangles_unwr[a]
            
            # smooth angles and call this 'curvature'
            sigma = int(np.round(0.0125*len(xs)))
            curvature = gaussian_filter1d(dangles_unwr, sigma = sigma, mode = 'wrap')
            
            # the minimum curvature is likely to be either the head or tail
            end_1 = int(np.where(curvature == np.min(curvature))[0][0])
            curvature_end_1 = curvature[end_1]
            if curvature_end_1 < -6 or curvature_end_1 > 0:
                import pdb; pdb.set_trace()
            
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

    @staticmethod
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
    

    
    
    ######################
    
    def create_summary_video(self, out_scale = 0.5):
        # setup video
        out_name = self.save_path + '\\' + os.path.splitext(self.vid_name)[0] \
            + '_tracking.avi'
        out_w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scale)
        out_h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scale)
        v_out = cv2.VideoWriter(out_name,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            self.vid.get(cv2.CAP_PROP_FPS), (out_w,out_h), 1)
        
        # setup font
        f_face = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 1.8
        f_thickness = 2
        f_color = (0,0,0)
        
        # loop through frames
        indices = np.linspace(0,self.num_frames-1,int(self.num_frames),dtype = 'uint16'); i = 0;
        for i in indices:
            print('Writing frame '+str(int(i+1))+' of '+str(int(self.num_frames)))
            
            # determine which tracks are present in the frame
            numbers = []
            centroids = []
            centerlines = []
            centerlines_unfixed = []
            centerline_flags = []
            centerline_flags_unfixed = []
            for w in range(len(self.centroids)):
                if i in np.arange(self.first_frames[w],self.first_frames[w]+len(self.centroids[w])):
                    numbers.append(w)
                    centroids.append(self.centroids[w][i-self.first_frames[w]])
                    if self.centerline_method != 'none':
                        centerlines.append(self.centerlines[w][i-self.first_frames[w]])
                        centerlines_unfixed.append(self.centerlines_unfixed[w][i-self.first_frames[w]])
                        centerline_flags.append(self.centerline_flags[w][i-self.first_frames[w]])
                        centerline_flags_unfixed.append(self.centerline_flags_unfixed[w][i-self.first_frames[w]])
                        
            # load frame
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_save = np.stack((img,img,img),2)
            
            for w in range(len(numbers)):
                text = str(numbers[w])
                text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
                text_pos = copy.copy(centroids[w]) # deepcopy avoids changing objs below
                text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                text_pos[1] = text_pos[1] + 30
                text_pos = tuple(np.uint16(text_pos))
                img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
                # cline
                if self.centerline_method != 'none':
                    pts = np.int32(centerlines[w][-1])
                    pts = pts.reshape((-1,1,2))
                    pts_unfixed = np.int32(centerlines_unfixed[w][-1])
                    pts_unfixed = pts_unfixed.reshape((-1,1,2))
                    if centerline_flags_unfixed[w] == 0:
                        img_save = cv2.polylines(img_save, pts, True, (255,0,0), 3)
                        img_save = cv2.circle(img_save, pts[0][0], 5, (255,0,0), -1)
                    elif centerline_flags_unfixed[w] == 1:
                        img_save = cv2.polylines(img_save, pts_unfixed, True, (0,0,255), 3)
                        #img_save = cv2.circle(img_save, pts_unfixed[0][0], 5, (0,0,255), -1)
                        if centerline_flags[w] == 2:
                            img_save = cv2.polylines(img_save, pts, True, (0,255,0), 3)
                            img_save = cv2.circle(img_save, pts[0][0], 5, (0,255,0), -1)
            img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
            
            v_out.write(img_save)
        print('DONE')
        v_out.release()


    def create_BW_video(self, out_scale = 1):
        # This method creates a BW video containing the thresholded masks
        # from application of the mask RCNN. This can be useful for developing
        # centerline-finding methods, etc. It could be improved by allowing
        # any stage of image processing to be output.
        out_name = self.save_path + '\\' + os.path.splitext(self.vid_name)[0] \
            + '_BW.avi'
        out_w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scale)
        out_h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scale)
        v_out = cv2.VideoWriter(out_name,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            self.vid.get(cv2.CAP_PROP_FPS), (out_w,out_h), 0)
        self.model, self.device = mrcnn.prepare_model(self.model_file)
        self.scale_factor = 0.5
        # get parameters (makes code below more readable)
        k_size = (round(self.parameters['k_sig']*3)*2+1,round(self.parameters['k_sig']*3)*2+1)
        k_sig = self.parameters['k_sig']
        bw_thr = self.parameters['bw_thr']
        area_bnds = self.parameters['area_bnds']
        
        # loop through frames
        indices = np.linspace(0,self.num_frames-1,int(self.num_frames),dtype = 'uint16'); i = 0;
        for i in indices:
            print('Writing frame '+str(int(i+1))+' of '+str(int(self.num_frames)))
                
            # load frame
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_save = np.stack((img,img,img),2)
            
            # inference with mRCNN, smooth, and threshold
            self.diff = mrcnn.segment_full_frame(img, self.model, self.device, self.scale_factor)
            smooth = cv2.GaussianBlur(self.diff,k_size,k_sig,cv2.BORDER_REPLICATE)
            thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)    
            img_save = bw

            if out_scale != 1:            
                img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
            
            v_out.write(img_save)
        
        v_out.release()

    
    def make_demo_frames(self,f):
        pass
        
    
class Intensity(Tracker):
    
    background_meth = 'max_merge'
    background_nframes = 10
    
    def __init__(self, vid_file):
        pass
    
    
    def segment_frame(self):
        pass
    
    

class MaskRCNN(Tracker):
    
    #model = blah
    
    def __init__(self, vid_file):
        pass


class TrackingQueue:
    
    def __init__(self):
        pass
    
    
    
if __name__ == '__main__':
    try:
        os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevents crash when using pytorch and matplotlib
        import matplotlib.pyplot as plt
        v1 = Tracker(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vids_cropped\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi')
        human_checked = False
        bkgnd_meth = 'max_merge'
        bkgnd_nframes = 10
        k_sig = 1.5
        bw_thr = 15
        area_bnds = (500,1000)
        d_thr = 15
        del_sz_thr = None
        um_per_pix = 8.5
        min_f = 50
        v1.set_parameters(human_checked, bkgnd_meth,bkgnd_nframes,k_sig,bw_thr,area_bnds,
                      d_thr,del_sz_thr,um_per_pix,min_f)
        #v1.track()
        
        f = 55
        img, diff, smooth, bw, bw_ws, final = v1.show_segmentation(f)
        #plt.imshow(final)
        
        #v2 = Tracker(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vids_cropped\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi')
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)