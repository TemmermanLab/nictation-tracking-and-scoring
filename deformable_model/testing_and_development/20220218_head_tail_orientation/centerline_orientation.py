# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:35:44 2022

@author: PDMcClanahan
"""
import os
import sys
import csv
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation')
import tracker_classes as Tracker
import head_tail_scoring_GUI as HT_GUI

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# methods
def load_centroids(centroids_file):
    xs = []
    ys = []
    first_frames = []
    
    with open(centroids_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                print(f'Column names are {", ".join(row)}')
            elif np.mod(row_count,2)==0:
                # ys.append(np.array(row[1:],dtype='float32'))
                ys.append(np.array(row[1:])[np.where(np.array(row[1:]) != '')[0]].astype('float32'))
            else:
                first_frames.append(int(row.pop(0)))
                # xs.append(np.array(row,dtype='float32'))
                xs.append(np.array(row)[np.where(np.array(row) != '')[0]].astype('float32'))
            row_count += 1
    
    # reshape
    centroids = []
    for t in range(len(xs)):
        centroids.append(np.swapaxes(np.vstack((xs[t],ys[t])),0,1))
    
    return centroids, first_frames


def load_centerlines(centerline_path):
    
    centerlines = []
    centerline_flags = []
    for file in os.listdir(centerline_path):
        if file.endswith(".csv"):
            with open(centerline_path+ "\\" + file) as csv_file:
                
                csv_reader = csv.reader(csv_file, delimiter=',')
                row_count = 0
                coord_list = []
                for row in csv_reader:
                    if row_count == 0:
                        row_count += 1
                    elif row_count == 1:
                        track_flags = np.array(row[1:],dtype=np.int16)
                        row_count +=1
                    else:
                        coord_list.append(np.array(row[1:],dtype='float32'))
                num_pts = int(len(coord_list)/2)
                num_frames = len(coord_list[0])
                
                # shape
                coord_list = np.array(coord_list,dtype = np.float32)
                centerlines_track = np.empty((num_frames,1, num_pts,2),
                                             dtype = np.float32)
                for f in range(num_frames):
                    centerlines_track[f,0,:,0] = coord_list[0:50,f]
                    centerlines_track[f,0,:,1] = coord_list[50:100,f]
                        
            centerlines.append(centerlines_track)
            centerline_flags.append(track_flags)                  
    
    return centerlines, centerline_flags


def load_end_angles(end_angles_file):
    '''Loads the end angles in <end_angles_file>, ignoring the header and worm
    numbers in the first row and first column. Returns a list of numpy arrays
    of the end angles for each frame that worm was tracked.'''
    
    end_angles = []
    with open(end_angles_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count > 0:
                angles = np.array(row[1:], dtype = np.float32)
                end_angles.append(angles)
            row_count += 1

    return end_angles
    


def create_vignettes(vid_name, vid_path, centroids, centerlines, first_frames):
    
    
    
    # created cropped videos of each tracked worm, centered around the
    # centroid, thus allowing for rapid loading during manual head / tail and
    # behavioral scoring
    
    # calculate the size of window to use based on maximal extent of tracked
    # worms
    extents = np.empty(0)
    for w in range(len(centerlines)):
        for f in range(np.shape(centerlines[w])[0]):
            extent = np.linalg.norm(np.float32(centerlines[w][f,0,0,:])-np.float32(centerlines[w][f,0,-1,:]))
            extents = np.append(extents,extent)
    halfwidth = int(np.max(extents)/1.7)       
    v_out_w = halfwidth*2+1; v_out_h = v_out_w
    
    # set up
    vid = cv2.VideoCapture(vid_path+'\\'+vid_name)
    
    save_path = vid_path + '\\' + os.path.splitext(vid_name)[0] + '_tracking\\vignettes'
    if not os.path.exists(save_path):
        print('Creating directory for tracking output: '+save_path)
        os.makedirs(save_path)
    is_color = 0
    
    # create vignettes of each worm
    for w in range(len(centerlines)):
        save_name = 'w'+str(w)+'.avi'
        v_out = cv2.VideoWriter(save_path+ '\\' +save_name, 
                                cv2.VideoWriter_fourcc('M','J','P','G'),
                                vid.get(cv2.CAP_PROP_FPS), (v_out_w,v_out_h),
                                is_color)
        first = first_frames[w]
        last = first_frames[w] + len(centroids[w])
    
        for f in range(first,last):
            msg = 'frame '+str(f-first+1)+' of '+str(last-first)+', track '+str(w)
            print(msg)
            vid.set(cv2.CAP_PROP_POS_FRAMES,f)
            frame = vid.read()[1]; frame = frame[:,:,0]
            canvas = np.uint8(np.zeros((np.shape(frame)[0]+halfwidth*2,np.shape(frame)[1]+halfwidth*2)))
            canvas[halfwidth:np.shape(frame)[0]+halfwidth,halfwidth:np.shape(frame)[1]+halfwidth] = frame
            centroid = np.uint16(np.round(centroids[w][f-first_frames[w]]))
            crop = canvas[centroid[1]:(centroid[1]+2*halfwidth),centroid[0]:(2*halfwidth+centroid[0])]
            v_out.write(crop)
        
        v_out.release()
        
    print('Done!')
    return halfwidth

def save_centroids(centroids, first_frames, save_path, 
                       save_name = 'centroids'):
        
    if not os.path.exists(save_path):
        print('Creating directory for centroids csv and other output: '+save_path)
        os.makedirs(save_path)
    
    save_file_csv = save_path + '\\' + save_name + '.csv'
    
    with open(save_file_csv, mode='w',newline="") as csv_file: 
        
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row = ['First Frame of Track','X and then Y Coordinates on Alternating Rows']
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
    
    

def save_centerlines(centerlines, centerline_flags, first_frames, save_path):

    if not os.path.exists(save_path + '\\centerlines'):
        print('Creating directory for centerlines csvs and other output: '+save_path+'\\centerlines')
        os.makedirs(save_path+'\\centerlines')
    
    for w in range(len(centerlines)):
        save_file_csv = save_path + '\\centerlines\\' + 'centerlines_worm_' + "{:06d}".format(w) + '.csv'
        
        with open(save_file_csv, mode='w',newline="") as csv_file: 
            
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # write row 1: frame number
            row = ['frame']
            for f in np.arange(first_frames[w],first_frames[w]+len(centerlines[w])):
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
                        row.append(str(round(float(centerlines[w][t][0,p,xy]),1)))
                    writer.writerow(row)   
            
    print("Centerlines saved in " + save_path + '\\centerlines')


def create_summary_video(vid_file, save_file, centroids, centerlines, first_frames, out_scale = 0.5):
    # setup video
    # out_name = self.save_path + '\\' + os.path.splitext(self.vid_name)[0] \
    #     + '_tracking.avi'
    vid = cv2.VideoCapture(vid_file)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * out_scale)
    out_h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * out_scale)
    v_out = cv2.VideoWriter(save_file,
        cv2.VideoWriter_fourcc('M','J','P','G'),
        vid.get(cv2.CAP_PROP_FPS), (out_w,out_h), 1)
    
    # setup font
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 1.8
    f_thickness = 2
    f_color = (0,0,0)
    
    # loop through frames
    indices = np.linspace(0,num_frames-1,int(num_frames),dtype = 'uint16'); i = 0;
    for i in indices:
        print('Writing frame '+str(int(i+1))+' of '+str(int(num_frames)))
        
        # determine which tracks are present in the frame
        numbers_f = []
        centroids_f = []
        centerlines_f = []
        centerlines_unfixed = []
        centerline_flags = []
        centerline_flags_unfixed = []
        for w in range(len(centroids)):
            if i in np.arange(first_frames[w],first_frames[w]+len(centroids[w])):
                numbers_f.append(w)
                centroids_f.append(centroids[w][i-first_frames[w]])
                centerlines_f.append(centerlines[w][i-first_frames[w]])
                # centerlines_unfixed.append(centerlines_unfixed[w][i-first_frames[w]])
                # centerline_flags.append(centerline_flags[w][i-first_frames[w]])
                # centerline_flags_unfixed.append(centerline_flags_unfixed[w][i-first_frames[w]])
                    
        # load frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_save = np.stack((img,img,img),2)
        
        #import pdb; pdb.set_trace()
        for w in range(len(numbers_f)):
            text = str(numbers_f[w])
            text_size = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
            text_pos = copy.copy(centroids_f[w]) # deepcopy avoids changing objs below
            text_pos[0] = text_pos[0]-text_size[0]/2 # x centering
            text_pos[1] = text_pos[1] + 30
            text_pos = tuple(np.uint16(text_pos))
            img_save = cv2.putText(img_save,text,text_pos,f_face,f_scale,f_color,f_thickness,cv2.LINE_AA)
            # cline
            
            pts = np.int32(centerlines_f[w][-1])
            pts = pts.reshape((-1,1,2))
            
            img_save = cv2.polylines(img_save, pts, True, (255,0,0), 3)
            img_save = cv2.circle(img_save, pts[0][0], 5, (255,0,0), -1)
        img_save = cv2.resize(img_save, (out_w,out_h), interpolation = cv2.INTER_AREA)
        
        v_out.write(img_save)
    print('DONE')
    v_out.release()


if __name__ == '__main__':
    try:
        
        # MANUAL SCORING OF H/T ORIENTATION
        
        # centroid_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centroids.csv'
        # centerline_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines'
        # vid_file = r'Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi'
        # vid_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline'
        # save_path_centroids = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        # save_path_centerlines = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        
        # centroid_file = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centroids.csv'
        # centerline_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines'
        # vid_file = r'Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi'
        # vid_path = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline'
        # save_path_centroids = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        # save_path_centerlines = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual'
        
        # centroids, first_frames = load_centroids(centroid_file)
        # centerlines, centerline_flags = load_centerlines(centerline_path)
        # # halfwidth = create_vignettes(vid_file, vid_path, centroids, centerlines, first_frames)
        # # print(halfwidth)

        # halfwidth = 188
        # centroids_man, centerlines_man, scores = HT_GUI.head_tail_inspector(
        #     vid_file, vid_path, centroids, centerlines, first_frames,
        #     halfwidth, scores = -1)
    
        # save_centroids(centroids_man, first_frames, save_path_centroids,
        #                     'centroids')
        # save_centerlines(centerlines_man, centerline_flags,
        #                       first_frames, save_path_centerlines)
        
        
        # # RELOAD MANUAL CENTERLINES
        man_centroid_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual\centroids.csv'
        man_centerline_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\centerlines_manual\centerlines'
        vid_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3.avi'
        profile_save_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\troubleshooting'
        save_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_manual.avi'
        end_1_angles_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\end_1_angles.csv' 
        end_2_angles_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vid_cropped_centerline\Sc_All_smell2_V2_ 21-09-17 14-51-41_crop_1_to_300_inc_3_tracking\end_2_angles.csv'
        
        centroids, first_frames = load_centroids(man_centroid_file)
        centerlines, centerline_flags = load_centerlines(man_centerline_path)
        scores = np.array([2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3,
                           2, 2, 1, 2, 2, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 
                           1, 1, 1, 1, 1, 1, 1, 2, 1],dtype = np.uint8)
        end_1_angles = load_end_angles(end_1_angles_file)
        end_2_angles = load_end_angles(end_2_angles_file)
        
        
        # remove censored centroids and centerlines
        indices = np.where(scores == 3)[0]
        for i in sorted(indices, reverse=True):
            del centerlines[i]
            del centroids[i]
            del first_frames[i]
            scores = np.delete(scores,i)
            del centerline_flags[i]
            del end_1_angles[i]
            del end_2_angles[i]
                
        # create_summary_video(vid_file, save_file, centroids, centerlines, first_frames, 0.5)
        # INDICATORS VS ORIENTATION
        
        
        
        # TOTAL MOVEMENT OF EACH END
        def dist(p1,p2):
            d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
            return d
        
        h_t_mov_all = []
        HT_MOV_IND = []
        for clines in centerlines:
            h_t_mov = [0,0]
            for f in range(len(clines)-1):
                h_t_mov[0] += dist(clines[f][0][0].astype(np.float32),
                             clines[f+1][0][0].astype(np.float32))
                h_t_mov[1] += dist(clines[f][0][-1].astype(np.float32),
                             clines[f+1][0][-1].astype(np.float32))
            h_t_mov_all.append(h_t_mov)
            HT_MOV_IND.append(h_t_mov[0]-h_t_mov[1])
        h_t_mov_all = np.array(h_t_mov_all)
        
        
        fig, axs = plt.subplots(1,1)
        axs.set_aspect('equal','box')
        axs.invert_yaxis()
        axs.plot(centerlines[8][:,0,0,0],centerlines[8][:,0,0,-1],'g-')
        axs.plot(centerlines[8][:,0,-1,0],centerlines[8][:,0,-1,-1],'r-')
        axs.set_title('Worm 8 head / tail path')
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        plt.show()
        
        
        fig, axs = plt.subplots(1,1)
        axs.set_aspect('equal','box')
        errors = 0
        for w in range(len(h_t_mov_all)):
            if h_t_mov_all[w][0] > h_t_mov_all[w][1]:
                axs.plot(h_t_mov_all[w][0],h_t_mov_all[w][1],'k.',markersize = 0+np.sqrt(len(centroids[w])))
            else:
                axs.plot(h_t_mov_all[w][0],h_t_mov_all[w][1],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                errors += 1
        
        axs.text(3000,100,'Error rate: '+str(errors / len(h_t_mov_all)))
        axs.plot([0,1000],[0,1000],'k:')
        axs.set_title('Total distance')
        axs.set_xlabel('tail')
        axs.set_ylabel('head')
        plt.show()
        
        
        
        # MIDPOINT DIRECTION
        N = np.shape(centerlines[0])[-2]
        fwd_mvnt = []
        bkwd_mvnt = []
        DIR_IND= []
        midpoint = round(N/2)
        for w in range(len(centerlines)):
            clines = centerlines[w]
            fwd_mvnt_w = 0
            bkwd_mvnt_w = 0
            for f in range(len(clines)-1):
                dir_vect = clines[f,0,midpoint-1]-clines[f,0,midpoint+1]
                dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                mov_vect = clines[f+1,0,midpoint]-clines[f,0,midpoint]
                mvnt_f = np.dot(dir_vect, mov_vect, out=None)
                if mvnt_f > 0:
                    fwd_mvnt_w += mvnt_f
                else:
                    bkwd_mvnt_w += abs(mvnt_f)
            fwd_mvnt.append(fwd_mvnt_w)
            bkwd_mvnt.append(bkwd_mvnt_w)
            DIR_IND.append(fwd_mvnt_w-bkwd_mov_w)
        
        fig, axs = plt.subplots(1,1)
        axs.set_title('Movement direction')
        axs.set_xlabel('forward movement (pix)')
        axs.set_ylabel('backward movement (pix)')
        axs.set_aspect('equal','box')
        errors = 0
        plt.plot([0,200],[0,200],'k:')
        for w in range(len(centerlines)):
            if fwd_mvnt[w] > bkwd_mvnt[w]:
                axs.plot(fwd_mvnt[w],bkwd_mvnt[w],'k.',markersize = 0+np.sqrt(len(centroids[w])))
            else:
                axs.plot(fwd_mvnt[w],bkwd_mvnt[w],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                errors += 1
        axs.text(-40,200,'Error rate: '+str(round(errors / len(centerlines),2)))
        plt.show()
        
        
        
        # LATERAL MOVEMENT OF THE HEAD AND TAIL
        end_1_lmvnt = []
        end_2_lmvnt = []
        LAT_MOV_IND = []
        for w in range(len(centerlines)):
            clines = centerlines[w]
            end_1_lmvnt_w = 0
            end_2_lmvnt_w = 0
            for f in range(len(clines)-1):
                # end 1
                dir_vect = clines[f,0,1]-clines[f,0,0]
                dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                mov_vect = clines[f+1,0,0]-clines[f,0,0]
                end_1_lmvnt_w += abs(np.cross(dir_vect, mov_vect))
                
                # end 2
                dir_vect = clines[f,0,-1]-clines[f,0,-2]
                dir_vect = dir_vect/np.sqrt(dir_vect[0]**2+dir_vect[1]**2)
                mov_vect = clines[f+1,0,-1]-clines[f,0,-1]
                end_2_lmvnt_w += abs(np.cross(dir_vect, mov_vect))
                
            end_1_lmvnt.append(end_1_lmvnt_w)
            end_2_lmvnt.append(end_2_lmvnt_w)
            LAT_MOV_IND.append(end_1_lmvnt_w-end_2_lmvnt_w)
            
        fig, axs = plt.subplots(1,1)
        axs.set_title('Lateral movement')
        axs.set_xlabel('head movement (pix)')
        axs.set_ylabel('tail mov. (pix)')
        axs.set_aspect('equal','box')
        errors = 0
        plt.plot([0,350],[0,350],'k:')
        for w in range(len(centerlines)):
            if end_1_lmvnt[w] > end_2_lmvnt[w]:
                axs.plot(end_1_lmvnt[w],end_2_lmvnt[w],'k.',markersize = 0+np.sqrt(len(centroids[w])))
            else:
                axs.plot(end_1_lmvnt[w],end_2_lmvnt[w],'r.',markersize = 0+np.sqrt(len(centroids[w])))
                errors += 1
        axs.text(2200,1,'Error rate: '+str(round(errors / len(centerlines),2)))

        plt.show()
        
        
        # POINTINESS
        fig, axs = plt.subplots(1,1)
        plt.plot([.1,.22],[.1,.22],'k:')
        axs.set_aspect('equal','box')
        axs.set_title('Sharpness of head and tail angles')
        axs.set_xlabel('head angle')
        axs.set_ylabel('tail angle')
        avg_head_angles = []
        avg_tail_angles = []
        POINT_IND = []
        errors = 0
        for w in range(len(end_1_angles)-1):
            if scores[w] == 1:
                avg_head_angle = np.mean(end_1_angles[w])
                avg_tail_angle = np.mean(end_2_angles[w])
            elif scores[w] == 2:
                avg_tail_angle = np.mean(end_1_angles[w])
                avg_head_angle = np.mean(end_2_angles[w])
            POINT_IND.append(avg_head_angle- avg_tail_angle)
            if avg_tail_angle < avg_head_angle:
                axs.plot(-avg_head_angle,-avg_tail_angle,'k.',markersize = 0+np.sqrt(len(centroids[w])))
            else:
                axs.plot(-avg_head_angle,-avg_tail_angle,'r.',markersize = 0+np.sqrt(len(centroids[w])))
                errors += 1
                if avg_head_angle > 0: break
        axs.text(.09,.27,'Error rate: '+str(round(errors / len(centerlines),2)))
        plt.show()
        from scipy import stats
        prob = stats.binom.cdf(errors , len(centerlines), 0.5)
        
        
        # INTENSITY
        avg_profiles = []
        vid = cv2.VideoCapture(vid_file)
        for w in range(len(centerlines)):
            shape = np.shape(centerlines[w])
            i_profile = np.empty((shape[2],shape[0]))
            frames = np.linspace(first_frames[w],
                                  first_frames[w]+len(centerlines[w])-1,
                                  len(centerlines[w]))
            for f, cline in enumerate(centerlines[w]):
                ret = vid.set(cv2.CAP_PROP_POS_FRAMES, frames[f])
                ret, img = vid.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                i = cline[0,:,0].astype(np.uint16)
                j = cline[0,:,1].astype(np.uint16)
                prof_frame = np.empty(np.shape(cline)[1])
                for p in range(len(prof_frame)):
                    prof_frame[p] = round(np.mean(img[j[p]-1:j[p]+2,i[p]-1:i[p]+2]))
                # i_profile[:,f] = img[j,i]
                i_profile[:,f] = prof_frame
            plt.imshow(i_profile)
            plt.show()
            filename = profile_save_path + '//worm_'+str(w)+'_intensity.bmp'
            try:
                ret = cv2.imwrite(filename, i_profile)
            except:
                pass
            av = np.mean(i_profile,-1)
            avg_profiles.append(av)
            
        # overall average profile
        avg_avg_profile = np.empty((50,len(centroids)))
        for p in range(len(avg_profiles)):
            avg_avg_profile[:,p] = avg_profiles[p]
        overall_avg = np.mean(avg_profiles,0)
        
        #np.sqrt(np.sum((avgp - overall_avg)**2)/len(p))
        fig, axs = plt.subplots(1,1)
        axs.plot(avg_avg_profile,'k-',linewidth = 0.2)
        axs.plot(np.flip(avg_avg_profile),'r-', linewidth = 0.2)
        axs.plot(overall_avg,'k-',linewidth = 1.5)
        axs.plot(np.flip(overall_avg),'r-', linewidth = 1.5)
        axs.set_title('Average centerline intensity + reverse')
        axs.set_xlabel('centerline point')
        axs.set_ylabel('intensity')
        #axs.set_aspect('equal','box')
        #errors = 0

        
        RMSdiffs = []
        RMSdiffs_rev = []
        INT_IND = []
        for p in range(len(avg_profiles)):
            prof = copy.copy(avg_profiles[p])
            RMSdiffs.append(np.sqrt(np.sum((prof - overall_avg)**2)/len(prof)))
            prof = np.flip(prof)
            RMSdiffs_rev.append(np.sqrt(np.sum((prof - overall_avg)**2)/len(prof)))
            INT_IND.append(RMSdiffs[-1]-RMSdiffs_rev[-1])
        
        fig, axs = plt.subplots(1,1)
        axs.set_aspect('equal','box')
        errors = 0
        for w in range(len(RMSdiffs)):
            if RMSdiffs[w] < RMSdiffs_rev[w]:
                axs.plot(RMSdiffs[w],RMSdiffs_rev[w],'k.')
            else:
                axs.plot(RMSdiffs[w],RMSdiffs_rev[w],'r.')
                errors += 1
            
        axs.text(20,1,'Error rate: '+str(round(errors / len(avg_profiles),2)))
        axs.plot([0,40],[0,40],'k:')
        axs.set_title('RMS difference from avg. intensity profile')
        axs.set_xlabel('correct')
        axs.set_ylabel('reversed')
        plt.show()
        
        
        # create arrays of the indicators for the forward and backward
        # orientation
        indicators = ['']
        
        
        # train a series of models
        
        
        # calculate the same indicators for a different video
        
        
        # test the performance of the combination of indicators on the test
        # video
        
        
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
    # load tracking information    
    

    # Create vignettes



# Load information needed for head / tail scoring



# manual head / tail scoring