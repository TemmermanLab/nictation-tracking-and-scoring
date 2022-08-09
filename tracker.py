# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:56:56 2021



# known issues and improvements:
    -automatically populate parameters based on scale and sample frame
    -load the parameters from another video
    -choose separate parameters for each video, or 'apply all'
    -change autoset tile_width and overlap according to scale or set manually
    -frame width in fix_centerlines should depend on image scale and worm type
    -summary video annotations should be scaled according to the size of the
     output frames
    -stitch together tracks by areal overlap instead of centerline distance
    -the code in show segmentation and find worms is largely redundant
    -there is a size change threshold but it is not; it would be useful for
     non-nictation tracking though as those worms will not change much in size
    -change how centerlines are structured internally (squeeze extra
     dimension, it should be frame, worm, [xs,ys])
    -check size change working

@author: PDMcClanahan
"""
import numpy as np
import os
import cv2
import copy
import csv
import tkinter as tk
from tkinter import simpledialog

# for debugging
import matplotlib.pyplot as plt
import time

# prevents crash when using pytorch and matplotlib
os.environ['KMP_DUPLICATE_LIB_OK']='True' 



# for mask RCNN tracking
import sys
sys.path.append(os.path.split(__file__)[0])
import mrcnn_module as mrcnn
import deformable_model as def_worm
import data_management_module as dm
import centerline_module as cm



class Tracker:
    
    num_vids = 0
    
    tshooting_outputs = [1]
    
    # for Steinernema, a max length of 360 works well for a 1.968 um / pixel
    # scale
    
    # print('WARNING: Using Steinernema metaparameters')
    # metaparameters = {
    #     'centerline_npts' : 50,
    #     'max_centerline_length' : 360,
    #     'max_centerline_angle' : 45,
    #     'edge_proximity_cutoff' : 10,
    #     'deformable_model_scale' : 0.25,
    #     }
    # model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\co'+\
    #         r'de\nictation\mask_R-CNN\Steinernema\20220127_full_frame_Sc_on'+\
    #         r'_udirt_4.pt'
    # model_file = os.path.split(__file__)[0] + \
    #     r'\mask_RCNN\Steinernema_mask_RCNN\20220127_full_frame_Sc_on_udirt_4.pt'
    # print('WARNING: Using S. carpocapsae metaparameters')
    
    metaparameters = {
        'centerline_method' : 'ridgeline',
        'centerline_npts' : 50,
        'max_centerline_length' : 750, # originally 268 pix, 690 is S. carpo. in um w/ 20% extra
        'max_centerline_angle' : 45,
        'edge_proximity_cutoff' : 10, # pix
        'deformable_model_scale' : 0.5,
        'summary_video_scale' : 1.0,
        'stitching_method' : 'overlap', # originally it was centroid distance
        }
    
    model_file = os.path.split(__file__)[0] + \
        r'\mask_RCNN\Celegans_mask_RCNN\20220331_full_frame_Ce_on_udirt_2.pt'
    print('WARNING: Using C. elegans metaparameters')
    
    model_scale = (960, 1280) # rows, cols
    
    size_factors = {
        'dauer' : 1.0,
        'IJ' : 1.2,
        'L3' : 1.0,
        'L4' : 1.2,
        'young adult' : 2.0,
        'adult' : 2.5,
        }
        

    def __init__(self, vid_file, segmentation_method = 'mask_RCNN'):
        
        
        self.vid_path, self.vid_name = os.path.split(vid_file) 
        self.segmentation_method = segmentation_method
        
        if self.segmentation_method == 'mask_RCNN':
            self.save_path = self.vid_path + '//' + self.vid_name[:-4] + \
                '_mRCNN_tracking'
            self.save_path_troubleshooting = self.save_path + \
                '//mRCNN_troubleshooting'
        elif self.segmentation_method == 'intensity':
            self.save_path = self.vid_path + '//' + self.vid_name[:-4] + \
                '_intensity_tracking'
            self.save_path_troubleshooting = self.save_path + \
                '//intensity_troubleshooting'
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(self.save_path_troubleshooting):
            os.mkdir(self.save_path_troubleshooting)
        
        if os.path.isfile(self.save_path+'\\tracking_parameters.csv'):
            print('Re-loading parameters')
            csv_filename = self.save_path+'\\tracking_parameters.csv'
            self.parameters = dm.load_parameter_csv(csv_filename)

        else:
            #self.auto_set_parameters(worm_type)
            self.parameters = {
            'human_checked' : False,
            'bkgnd_meth' : 'max_merge',
            'bkgnd_nframes' : 10,
            'k_sig' : 6.5, # um, default, close to original 1.5 pix
            'bw_thr' : 10,
            'area_bnds' : (3700 , 16650), # sq um for C. elegans dauers
            'd_thr' : 430, # um, based on 100 um
            'del_sz_thr' : '',
            'um_per_pix' : 4.3, # default based on data in C. elegans dataset
            'min_f' : 300
            }
            #self.save_params_csv('tracking_parameters')
        
        
        self.vid = cv2.VideoCapture(self.vid_path+'//'+self.vid_name)
        self.num_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.segmentation_method == 'intensity':
            self.background = self.get_background(self.vid,
                                            self.parameters['bkgnd_nframes'])
        else:
            self.background = None
        self.dimensions = (int(self.vid.get(4)),int(self.vid.get(3)))
        # assumes same aspect ratio
        self.scale_factor = self.model_scale[0]/self.dimensions[0]
        
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
        
    # wrapper called by tracking GUI
    def save_params(self):
        
        
        # if self.segmentation_method == 'mask_RCNN':
        #     self.save_path = self.vid_path + '//' + self.vid_name[:-4] + \
        #         '_mRCNN_tracking'
        #     self.save_path_troubleshooting = self.save_path + \
        #         '//mRCNN_troubleshooting'
        # elif self.segmentation_method == 'intensity':
        #     self.save_path = self.vid_path + '//' + self.vid_name[:-4] + \
        #         '_intensity_tracking'
        #     self.save_path_troubleshooting = self.save_path + \
        #         '//intensity_troubleshooting'
        
        
        # save_path = self.vid_path + '\\' \
        #     + os.path.splitext(self.vid_name)[0] + '_tracking'
        # dm.save_params_csv(self.parameters, save_path, 'tracking_parameters')
        
        dm.save_params_csv(self.parameters, self.save_path, 'tracking_parameters')
    
    
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
        
    
    
    
    # creates images for display in parameter GUI
    def show_segmentation(self, f=0):
        
        
        # convert parameters to pixels for practical use
        k_sig = self.parameters['k_sig'] * (1/self.parameters['um_per_pix'])
        k_size = (round(k_sig*3)*2+1,
                  round(k_sig*3)*2+1)
        area_bnds = np.array(self.parameters['area_bnds']) * \
            (1/(self.parameters['um_per_pix']**2))
        max_angle = self.metaparameters['max_centerline_angle']
        max_length = self.metaparameters['max_centerline_length'] * \
            (1/self.parameters['um_per_pix'])
        bw_thr = self.parameters['bw_thr']
        
        
        # set up mask RCNN if needed
        if self.segmentation_method == 'mask_RCNN' and \
                                   'self.model_file' not in locals():
            self.model, self.device = mrcnn.prepare_model(self.model_file)
            self.param_gui_f = -1 # tracks for which 'diff' was calculated
        
        
        # read in frame f
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        # make bw image
        if self.segmentation_method == 'intensity':
            self.diff = (np.abs(img.astype('int16') - \
                        self.background.astype('int16'))).astype('uint8')
        elif self.segmentation_method == 'mask_RCNN':
            if self.param_gui_f == f: # do not recalculate if not necessary
                pass
            else:
                self.diff = mrcnn.segment_full_frame(img, self.model,
                                            self.device, self.scale_factor)
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
                bw_w = copy.copy(cc[1][cc[2][cc_i,1]:cc[2][cc_i,1]+ \
                    cc[2][cc_i,3],cc[2][cc_i,0]:cc[2][cc_i,0]+cc[2][cc_i,2]])
                bw_w[np.where(bw_w == cc_i)]=255
                bw_w[np.where(bw_w!=255)]=0
                cline = np.float32(cm.find_centerline(bw_w)[0])
                cline_flags.append(cm.flag_bad_centerline(cline, 
                                            max_length, max_angle))
                cline[:,0] += cc[2][cc_i][0]; cline[:,1] += cc[2][cc_i][1] 
                clines_f.append(copy.copy(cline))
                
        
        # setup overlay text based on image size
        font_factor = self.dimensions[1]/720
        f_face = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = .5 * font_factor
        f_thickness = round(2 * font_factor)
        f_color = (0,0,0)
        linewidth = round(1 * font_factor)
        linewidth_scale = round(2 * font_factor)
        offset = round(70 * font_factor)
        
        # create 'final' image showing identified worms
        final_HSV = cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),
                                 cv2.COLOR_BGR2HSV)  
        
        # add red shading to all bw blobs
        final_HSV[:,:,0][np.where(bw==255)] = 120 # set hue (color)
        # set saturation (amount of color, 0 is grayscale)
        final_HSV[:,:,1][np.where(bw==255)] = 80 
    
        # change shading to green for those that are within the size bounds
        final_HSV[:,:,0][np.where(bw_ws==255)] = 65 # set hue (color)
        
        # convert image to BGR (the cv2 standard)
        final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
        
        # could also outline blobs within size bounds
        # final = cv2.cvtColor(final_HSV,cv2.COLOR_HSV2BGR)
        # contours, hierarchy = cv2.findContours(bw_ws, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(final, [contours[0]], 0, (0, 255, 0), 1) #drawing contours
        
        # label blobs detected as worms with centerline
        for trk in range(np.shape(centroids)[0]):
            # cline
            pts = np.int32(clines_f[trk])
            pts = pts.reshape((-1,1,2))
            final = cv2.polylines(final, pts, True, (0,255,0), linewidth)
        
        # label the size of all blobs
        for cc_i in cc_is[1:]:
            cc_sz = cc[2][cc_i][4]
            text = str(round(cc_sz*(self.parameters['um_per_pix']**2)))
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            text_pos = copy.copy(cc[3][cc_i]) 
            text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
            text_pos[1] = text_pos[1] + 30
            text_pos = tuple(np.uint16(text_pos))
            if cc_sz > area_bnds[0] and cc_sz < area_bnds[1]:
                final = cv2.putText(final,text,text_pos,f_face,f_scale,
                                    f_color,f_thickness,cv2.LINE_AA)
            else:
                final = cv2.putText(final,text,text_pos,f_face,f_scale,
                                    (50,50,50),f_thickness,cv2.LINE_AA)
     
        # show the distance threshold
        if self.parameters['d_thr'] is not None:
            d_thr = np.round(self.parameters['d_thr'] * \
                (1/self.parameters['um_per_pix'])).astype(np.int32)
            text = 'd='+str(self.parameters['d_thr'])+' um'
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            pt1 = [np.shape(img)[1]-offset,np.shape(img)[0]-offset]
            pt2 = [pt1[0]-d_thr,pt1[1]]
            text_pos = np.array((((pt1[0]+pt2[0])/2,pt1[1])),dtype='uint16')
            text_pos[0] = text_pos[0] - text_size[0]/2 # x centering 
            text_pos[1] = text_pos[1] - 5 # y offset
            final = cv2.polylines(final, np.array([[pt1,pt2]]), True, 
                                 (0,0,255), linewidth)
            final = cv2.putText(final,text,tuple(text_pos),f_face,f_scale,
                                (0,0,255),f_thickness,cv2.LINE_AA)
            del pt1, pt2
        
        return img, self.diff, smooth, bw, bw_ws, final
        
        
    def track(self,fix_centerlines = True):
        
        out_scale = self.metaparameters['summary_video_scale']
        
        # set up model if using mask RCNN
        if self.segmentation_method == 'mask_RCNN':
            self.model, self.device = mrcnn.prepare_model(self.model_file)
        
        
        # set up loop, vars to hold centroid
        self.centroids_raw = []
        self.first_frames_raw = []
        if self.metaparameters['centerline_method']  != 'none':
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
        track_loop_times = []
        
        indices = np.linspace(0,self.num_frames-1,int(self.num_frames))
        
        for i in indices:
            if i == -1:#indices[21]:
                self.debug = True
            else:
                self.debug = False
            
            start_time = time.time()
            print('Finding worms in frame '+str(int(i+1))+' of '+ 
                  str(int(self.num_frames)))
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,img = self.vid.read(); img = cv2.cvtColor(img,
                                                          cv2.COLOR_BGR2GRAY)
            
            centroids_frame, centerlines_frame, centerline_flags_frame, \
                    angles_end_1_frame, angles_end_2_frame = \
                        self.find_worms(img, self.segmentation_method, self.metaparameters, self.parameters, 
                                    self.background, self.model, self.device, self.scale_factor)
            
            self.stitch_centroids(centroids_frame, centerlines_frame,
                                  centerline_flags_frame, angles_end_1_frame,
                                  angles_end_2_frame, i)
            
            track_loop_times.append(time.time()-start_time)
        
        #import pdb; pdb.set_trace()
        plt.plot(track_loop_times,'k.')
        average = round(np.sum(track_loop_times)/len(track_loop_times),2)
        plt.title('Tracking Loop Timing ('+str(average)+' s per frame)')
        plt.xlabel('Frame')
        plt.ylabel('Time (s)')
        plt.savefig(self.save_path_troubleshooting+'//'+'tracking_timing.png')
        plt.show()
        

        # cleanup
        self.remove_short_tracks()
        if fix_centerlines:
            dm.save_centerlines(self.centerlines, self.centerline_flags,
                                  self.first_frames, self.save_path, 
                                  'centerlines_unfixed')
            self.fix_centerlines()
        
        self.centerlines, self.angles_end_1, self.angles_end_2 = \
            cm.rectify_centerlines(self.centerlines, self.angles_end_1,
                                   self.angles_end_2)
        
        # save results
        dm.save_centroids(self.centroids, self.first_frames, self.save_path,
                            'centroids')
        dm.save_centerlines(self.centerlines, self.centerline_flags,
                              self.first_frames, self.save_path)
        dm.save_end_angles(self.angles_end_1, self.save_path, '1')
        dm.save_end_angles(self.angles_end_2, self.save_path, '2')
        
        # make tracking video
        self.create_summary_video(out_scale)
        
        


    @staticmethod
    def get_background(vid, bkgnd_numf):
        print('Calculating background image...')
        
        numf = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        
        inds = np.round(np.linspace(0,numf-1,
                        bkgnd_numf)).astype(int)
             
        for i in inds:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            if i == inds[0]:
                ret,stack = vid.read()
                stack = cv2.cvtColor(stack, cv2.COLOR_BGR2GRAY)
                stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1)) 
            else:
                ret,img = vid.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.reshape(img,(img.shape[0],img.shape[1],1)) 
                stack = np.concatenate((stack, img), axis=2)
                stack = np.amax(stack,2)
                stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1)) 
        
        return np.squeeze(stack)
        
    
    @staticmethod
    def find_worms(img, segmentation_method, metaparameters, parameters,
                   background = None, model = None, device = None, 
                   scale_factor = None, return_bw = False):
        # debug, max_centerline_angle, max_centerline_length, k_sig, bw_thr, area_bnds, segmentation_method, model, device, scale_factor, background, edge_proximity_cutoff, centerline_method
        debug = False # for find_centerline, do not comment out or will crash
        
        
        # convert parameters to pixels for practical use
        k_sig = parameters['k_sig'] * (1/parameters['um_per_pix'])
        k_size = (round(k_sig*3)*2+1,
                  round(k_sig*3)*2+1)
        area_bnds = np.array(parameters['area_bnds']) * \
            (1/parameters['um_per_pix']**2)
        max_angle = metaparameters['max_centerline_angle']
        max_length = metaparameters['max_centerline_length'] * \
            (1/parameters['um_per_pix'])
        bw_thr = parameters['bw_thr']
        
        
        
        if segmentation_method == 'intensity':
            diff = (np.abs(img.astype('int16') - \
                           background.astype('int16'))).astype('uint8')
        elif segmentation_method == 'mask_RCNN':
            diff = mrcnn.segment_full_frame(img, model, device, scale_factor)
        
        smooth = cv2.GaussianBlur(diff,k_size,k_sig,cv2.BORDER_REPLICATE)
        thresh,bw = cv2.threshold(smooth,bw_thr,255,cv2.THRESH_BINARY)
        
        # cc: # objs, labels, stats, centroids
        #  -> stats: left, top, width, height, area
        cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
        cc_map = np.uint8(cc[1]); 
        cc_is = np.linspace(0,cc[0]-1,cc[0]).astype(int)

        # eliminate objects that are too big, too small, or touch the boundary
        bw_ws = np.zeros(np.shape(bw),dtype = 'uint8')
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

                if np.min(obj_inds_r) <= metaparameters['edge_proximity_cutoff'] or np.min(obj_inds_c) <= metaparameters['edge_proximity_cutoff']:
                    hits_edge = True
                elif np.max(obj_inds_r) >= np.shape(cc[1])[0]-1-metaparameters['edge_proximity_cutoff'] or np.max(obj_inds_c) >= np.shape(cc[1])[1]-1-metaparameters['edge_proximity_cutoff']:
                    hits_edge = True
                    
                if hits_edge is False:
                    centroids.append(copy.deepcopy(cc[3][cc_i]))
                    bw_ws[np.where(cc_map==cc_i)] = 255
                    
                    # find the centerline
                    if metaparameters['centerline_method'] != 'none':
                        bw_w = copy.copy(cc[1][cc[2][cc_i,1]-1:cc[2][cc_i,1]-1+cc[2][cc_i,3]+2,cc[2][cc_i,0]-1:cc[2][cc_i,0]-1+cc[2][cc_i,2]+2])
                        bw_w[np.where(bw_w == cc_i)]=255
                        bw_w[np.where(bw_w!=255)]=0
                        
                        try:
                            centerline, angle_end_1, angle_end_2 = \
                                cm.find_centerline(bw_w,metaparameters['centerline_method'], debug)
                            centerline = np.float32(centerline)
                            centerline_flag = cm.flag_bad_centerline(centerline,
                                                            max_length, max_angle)
                            centerline[:,0] += cc[2][cc_i][0]
                            centerline[:,1] += cc[2][cc_i][1]
                            centerline = centerline[np.newaxis,...]
                        except:
                            centerline = None
                            centerline_flag = 1
                            angle_end_1 = None
                            angle_end_2 = None
                            
                        centerlines.append(copy.copy(centerline))
                        
                        centerline_flags.append(centerline_flag)
                        
                        angles_end_1.append(angle_end_1)
                        angles_end_2.append(angle_end_2)
                        
        # re-arrange centroids
        if metaparameters['centerline_method'] != 'none' and not return_bw:
            return centroids, centerlines, centerline_flags, angles_end_1, angles_end_2
        elif metaparameters['centerline_method'] != 'none' and return_bw:
            return centroids, centerlines, centerline_flags, angles_end_1, angles_end_2, bw_ws
        else:
            return centroids
    
    

    
    def stitch_centroids(self, centroids_frame, centerlines_frame, 
                         centerline_flags_frame, angles_end_1_frame, 
                         angles_end_2_frame, f):
        
        # convert d_thr to pixels
        d_thr = self.parameters['d_thr'] + (1/self.parameters['um_per_pix'])
        
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
                if self.metaparameters['centerline_method'] != 'none':
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
                if min_dist < d_thr:
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
                if self.metaparameters['centerline_method'] != 'none':
                    self.centerlines_raw[prev_obj_inds[pair_list[i][0]]].append(centerlines_frame[pair_list[i][1]])
                    self.centerline_flags_raw[prev_obj_inds[pair_list[i][0]]].append(centerline_flags_frame[pair_list[i][1]])
                    self.angles_end_1_raw[prev_obj_inds[pair_list[i][0]]].append(angles_end_1_frame[pair_list[i][1]])
                    self.angles_end_2_raw[prev_obj_inds[pair_list[i][0]]].append(angles_end_2_frame[pair_list[i][1]])
            
            # newly tracked objects
            for i in range(len(centroids_frame)):
                if i not in tracked_inds:
                    self.centroids_raw.append([centroids_frame[i]])
                    self.first_frames_raw.append(int(f))
                    if self.metaparameters['centerline_method'] != 'none':
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
        
        k_sig = self.parameters['k_sig'] * (1/self.parameters['um_per_pix'])
        k_size = (round(k_sig*3)*2+1,
                  round(k_sig*3)*2+1)
        max_angle = self.metaparameters['max_centerline_angle']
        max_length = self.metaparameters['max_centerline_length'] * \
            (1 / self.parameters['um_per_pix'])
        
        self.centerlines_unfixed = copy.deepcopy(self.centerlines)
        self.centerline_flags_unfixed = copy.deepcopy(self.centerline_flags)
        fw = 200 # frame_width, should be changed to depend on may worm size
        k_size = (round(self.parameters['k_sig']*3)*2+1,
                  round(self.parameters['k_sig']*3)*2+1)
        
        
        flags_new = copy.deepcopy(self.centerline_flags)
        

        # determine the number of centerlines that need to be fixed, as this
        # is a time-consuming step
        num_flagged = 0
        num_fixed = 0
        for i in range(len(self.centerline_flags)): 
            num_flagged += len(np.where(self.centerline_flags[i])[0] == 1)
        
        fix_centerline_times = []
        
        # fix centerlines worm by worm
        for w in range(len(self.centerlines)):
            #if w == 15: import pdb; pdb.set_trace()
            flags = np.array(self.centerline_flags[w]) # new flags after fixing
            if len(np.where(flags==1)[0]) == len(flags):
                print('Worm ' + str(w) + ' has no non-flagged centerlines')
            else:
                flags_prog = copy.copy(flags) # copy for distance transform purposes
                dists_orig = cm.dist_trans(flags_prog) # for reference
                # fix centerlines in order of closeness to a good centerline
                while len(np.where(flags_prog==1)[0]) > 0:
                    
                    start_time = time.time()
                    print('Fixing flagged centerline '+ str(num_fixed+1) + \
                          ' of ' + str(num_flagged))
                    
                    # choose a centerline to correct based on proximity to a 
                    # non-flagged or fixed centerline, using the distance from
                    # a non-flagged centerline to break ties
                    dists_now = cm.dist_trans(flags_prog)
                    potential_next_fs = np.where(dists_now == 1)[0]
                    f = potential_next_fs[np.where(dists_orig[potential_next_fs] == np.min(dists_orig[potential_next_fs]))[0][0]]
                    # f = np.where(dists_now == 1)[0][0]
                    
                    # choose a moving centerline from the adjacent centerlines 
                    # 1st priority is given to adjacent centerlines that were
                    # never flagged (i.e. where dists_orig == 0)
                    # 2nd priority is given to adjacent centerlines that were
                    # flagged and fixed, and are closer to a non-flagged
                    # centerline than the other adjacent centerline
                    # 3rd priority is given to the only adjacent fixed
                    # centerline
                    # 4th priority is the cases of the first and last frames,
                    # in which there is no alternative
                    wtf = True
                    if f > 0 and f < len(self.centerlines[w])-1:
                        if dists_orig[f-1] == 0:
                            moving_centerline = self.centerlines[w][f-1]
                            wtf = False
                        elif dists_orig[f+1] == 0:
                            moving_centerline = self.centerlines[w][f+1]
                            wtf = False
                        elif dists_now[f-1] == 0 and dists_now[f+1] == 0:
                            if dists_orig[f-1] <= dists_orig[f+1]:
                                moving_centerline = self.centerlines[w][f-1]
                                wtf = False
                            elif dists_orig[f-1] > dists_orig[f+1]:
                                moving_centerline = self.centerlines[w][f+1]
                                wtf = False
                        elif dists_now[f-1] == 0:
                            moving_centerline = self.centerlines[w][f-1]
                            wtf = False
                        else: # dists_now[f+1] == 0
                            moving_centerline = self.centerlines[w][f+1]
                            wtf = False
                    elif f == 0:
                        moving_centerline = self.centerlines[w][f+1]
                        wtf = False
                    else:
                        moving_centerline = self.centerlines[w][f-1]
                        wtf = False
                    
                    if wtf:
                        import pdb; pdb.set_trace()
                    # if f > 0 and flags_prog[f-1] == 0: # frame before is OK
                    #     moving_centerline = self.centerlines[w][f-1]
                    # elif f < len(self.centerlines[w])-1 and flags_prog[f+1] == 0: # frame after is OK
                    #     moving_centerline = self.centerlines[w][f+1]
                    
                    # retrieve the target image (bw segmentation of the worm that
                    # caused the flagged centerline)
                    unshifted_f = self.first_frames[w] + f
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, unshifted_f)
                    ret,img = self.vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if self.segmentation_method == 'intensity':
                        diff = (np.abs(img.astype('int16') - 
                            self.background.astype('int16'))).astype('uint8')
                    elif self.segmentation_method == 'mask_RCNN':
                        diff = mrcnn.segment_full_frame(img, self.model,
                                                        self.device,
                                                        self.scale_factor)
                    smooth = cv2.GaussianBlur(diff,k_size, k_sig,
                                              cv2.BORDER_REPLICATE)
                    thresh,bw = cv2.threshold(smooth,
                            self.parameters['bw_thr'],255,cv2.THRESH_BINARY)
                    height, width = bw.shape[:2]
                    mp = round(self.metaparameters['centerline_npts']/2)
                    
                    # eliminate other nearby segmentations using floodfill
                    mask = np.zeros((height + 2, width + 2), np.uint8)
                    floodfilled = False
                    
                    # check if points in moving_centerline are in an ROI
                    for p in range(np.shape(moving_centerline)[1]):
                        # in case a point near the edge rounds out of bounds
                        try: 
                            if bw[moving_centerline[0][p][1].astype(np.int16),
                                moving_centerline[0][p][0].astype(np.int16)] \
                                == 255:
                                cv2.floodFill(bw, mask, (
                                    moving_centerline[0][p][0].astype(np.int16),
                                    moving_centerline[0][p][1].astype(np.int16)),
                                    127)
                                floodfilled = True; break
                        except:
                            pass
                    
                    # failing that, look for the foreground point in bw
                    # closest to the midpoint of the moving centerline
                    if not floodfilled:
                        foreground_points = np.array(np.where(bw == 255))
                        distances = np.empty(np.shape(foreground_points)[1])
                        midpoint = moving_centerline[0][
                            mp]
                        midpoint = np.flip(midpoint)
                        for p in range(len(distances)):
                            distances[p] = np.linalg.norm(
                                midpoint-foreground_points[:,p])
                        flood_seed_i = np.where(distances==np.min(distances))[0][0]
                        flood_seed = tuple(foreground_points[:,flood_seed_i].astype(np.int16))
                        flood_seed = (flood_seed[1],flood_seed[0])
                        cv2.floodFill(bw, mask, flood_seed, 127)
                        floodfilled = True
                    
                    if floodfilled:
                        bw[np.where(bw==255)]=0
                        bw[np.where(bw==127)]=255
                        
                    # add a frame
                    bw_framed = np.zeros((np.shape(bw)[0]+2*fw,
                                          np.shape(bw)[1]+2*fw),
                                         dtype = 'uint8')
                    bw_framed[fw:-fw,fw:-fw] = bw
                    target_coords = moving_centerline[0][mp].astype(np.int16)+200
                    target_image = bw_framed[(target_coords[1]-fw):(target_coords[1]+fw),(target_coords[0]-fw):(target_coords[0]+fw)]
                    moving_centerline_shifted = copy.copy(moving_centerline)
                    moving_centerline_shifted[0,:,0] = moving_centerline[0,:,0] - moving_centerline[0][mp][0] + 200
                    moving_centerline_shifted[0,:,1] = moving_centerline[0,:,1] - moving_centerline[0][mp][1] + 200
                    moving_centerline_shifted = np.swapaxes(np.squeeze(moving_centerline_shifted,0),1,0)
    
                    # import matplotlib.pyplot as plt
                    # plt.imshow(target_image, cmap = 'gray')
                    # plt.plot(moving_centerline_shifted[0,:],moving_centerline_shifted[1,:],'r--')
                    # plt.show()
                    
                    # downscale the target image and moving centerline for 
                    # improved speed
                    scale = self.metaparameters['deformable_model_scale']
                    width = int(target_image.shape[1] * scale)
                    height = int(target_image.shape[0] * scale)
                    dim = (width, height)
                    target_image_scaled = cv2.resize(target_image, dim,
                                            interpolation = cv2.INTER_AREA)
                    thr, target_image_scaled = cv2.threshold(
                        target_image_scaled,127,255,cv2.THRESH_BINARY)
                    moving_centerline_shifted_scaled = \
                        moving_centerline_shifted * scale
                    
                    # #debugging
                    # if target_image_scaled[0,0] == 255:
                    #     import pdb; pdb.set_trace()
                    
                    
                    # fit the gravitational spline model to the target image
                    #if f == 9: sdfasd
                    grav_model = def_worm.Gravitational_model(
                        moving_centerline_shifted_scaled, target_image_scaled)
                    new_centerline = grav_model.fit(False, False, None) / scale
                    
                    # initialize a deformable model

                    # deformable_model = def_worm.Eigenworm_model()
                    # deformable_model.set_centerline(moving_centerline_shifted)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(deformable_model.bw_image, cmap = 'gray'); plt.show()
                    
                    # # fit the deformable model to the segmentation by gradient descent
                    # n_iter = 100
                    # lr = [20,20,20,.5,3,3,3,3,3]
                    # grad_step = [1,1,1,1,0.1,0.1,0.1,0.1,0.1]
                    # max_step = [2,2,2,0.02,0.1,0.1,0.1,0.1,0.1]
                    # save_dir = self.save_path_troubleshooting
                    # show = False
                    # vid = False
                    # optimizer = def_worm.Gradient_descent(deformable_model,
                    #                                       target_image,
                    #                                       n_iter, lr,
                    #                                       grad_step, max_step,
                    #                                       save_dir, show, vid)
                    # optimizer.run()
                    # # plt.imshow(deformable_model.bw_image, cmap = 'gray'); plt.show()
    
                    # # check the new centerline
                    
                    # # save the new centerline and change the flag value to 2
                    # new_centerline = deformable_model.centerline
                    new_centerline = new_centerline.swapaxes(0,1)
                    new_centerline = np.expand_dims(new_centerline,0)
                    new_centerline[0,:,0] += moving_centerline[0][mp].astype(np.int16)[0]-200
                    new_centerline[0,:,1] += moving_centerline[0][mp].astype(np.int16)[1]-200
                    self.centerlines[w][f] = new_centerline
                    new_centerline_flag = cm.flag_bad_centerline(new_centerline[0], max_length, max_angle)
                    if new_centerline_flag:
                        flags[f] = 3
                    else:
                        flags[f] = 2
                    flags_prog[f] = 0
                    
                    # check the new centerline and change the flag to 2 (fixed) if it passes
                    num_fixed +=1
                    
                    fix_centerline_times.append(time.time()-start_time)
                    
                self.centerline_flags[w] = copy.deepcopy(flags.tolist())
        
        plt.plot(fix_centerline_times,'k.')
        average = round(np.sum(fix_centerline_times)/len(fix_centerline_times)
                        ,2)
        fixes_per_frame = round(len(fix_centerline_times)/self.num_frames,2)
        plt.title('Centerline Fixing Timing ('+str(average)+' s per fix)')
        plt.xlabel('Count (' + str(fixes_per_frame) + ' fixes per frame)')
        plt.ylabel('Time (s)')
        plt.savefig(self.save_path_troubleshooting+'//'+
                    'centerlin_fixing_timing.png')
        plt.show()



    def orient_head_tail(self):
        
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
    

    
    def create_summary_video(self, out_scale = 1.0):
        # setup video
        out_name = self.save_path + '\\' + \
            os.path.splitext(self.vid_name)[0] + '_tracking.avi'
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
        indices = np.linspace(0,self.num_frames-1,int(self.num_frames),
                              dtype = 'uint16'); i = 0;
        for i in indices:
            print('Writing frame '+str(int(i+1))+' of '+ \
                  str(int(self.num_frames)))
            
            # determine which tracks are present in the frame
            numbers = []
            centroids = []
            centerlines = []
            centerlines_unfixed = []
            centerline_flags = []
            centerline_flags_unfixed = []
            for w in range(len(self.centroids)):
                if i in np.arange(self.first_frames[w],
                                 self.first_frames[w]+len(self.centroids[w])):
                    numbers.append(w)
                    centroids.append(
                        self.centroids[w][i-self.first_frames[w]])
                    if self.metaparameters['centerline_method'] != 'none':
                        centerlines.append(
                            self.centerlines[w][i-self.first_frames[w]])
                        centerlines_unfixed.append(
                            self.centerlines_unfixed[w][
                                i-self.first_frames[w]])
                        centerline_flags.append(
                            self.centerline_flags[w][i-self.first_frames[w]])
                        centerline_flags_unfixed.append(
                            self.centerline_flags_unfixed[w][
                                i-self.first_frames[w]])
                        
            # load frame
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,img = self.vid.read(); img = cv2.cvtColor(img, 
                                                          cv2.COLOR_BGR2GRAY)
            img_save = np.stack((img,img,img),2)
            
            for w in range(len(numbers)):
                text = str(numbers[w])
                text_size = cv2.getTextSize(text, f_face, f_scale,
                                            f_thickness)[0]
                text_pos = copy.copy(centroids[w]) # avoid changing objs below
                text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
                text_pos[1] = text_pos[1] + 30
                text_pos = tuple(np.uint16(text_pos))
                img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,
                                       f_color,f_thickness,cv2.LINE_AA)
                # cline
                if self.metaparameters['centerline_method'] != 'none':
                    pts = np.int32(centerlines[w][-1])
                    pts = pts.reshape((-1,1,2))
                    pts_unfixed = np.int32(centerlines_unfixed[w][-1])
                    pts_unfixed = pts_unfixed.reshape((-1,1,2))
                    if centerline_flags_unfixed[w] == 0:
                        img_save = cv2.polylines(img_save, pts, True,
                                                 (255,0,0), 3)
                        img_save = cv2.circle(img_save, pts[0][0], 5, 
                                              (255,0,0), -1)
                    elif centerline_flags_unfixed[w] == 1:
                        img_save = cv2.polylines(img_save, pts_unfixed, True, 
                                                 (0,0,255), 3)
                        # img_save = cv2.circle(img_save, pts_unfixed[0][0], 5, 
                        #                       (0,0,255), -1)
                        if centerline_flags[w] == 2:
                            img_save = cv2.polylines(img_save, pts, True,
                                                     (0,255,0), 3)
                            img_save = cv2.circle(img_save, pts[0][0], 5,
                                                  (0,255,0), -1)
                        elif centerline_flags[w] == 3:
                            img_save = cv2.polylines(img_save, pts, True,
                                                     (0,255,255), 3)
                            img_save = cv2.circle(img_save, pts[0][0], 5,
                                                  (0,255,255), -1)
                            
            img_save = cv2.resize(img_save, (out_w,out_h),
                                  interpolation = cv2.INTER_AREA)
            
            v_out.write(img_save)
        print('DONE')
        v_out.release()


    def create_BW_video(self, out_scale = 1):
        # This method creates a BW video containing the thresholded masks
        # from application of the mask RCNN. This can be useful for developing
        # centerline-finding methods, etc. It could be improved by allowing
        # any stage of image processing to be output.
        out_name = self.save_path + '\\' + os.path.splitext(self.vid_name)[0]\
            + '_BW.avi'
        out_w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scale)
        out_h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scale)
        v_out = cv2.VideoWriter(out_name,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            self.vid.get(cv2.CAP_PROP_FPS), (out_w,out_h), 0)
        self.model, self.device = mrcnn.prepare_model(self.model_file)
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