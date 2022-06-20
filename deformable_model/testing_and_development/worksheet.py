# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:08:55 2021

@author: Temmerman Lab
"""

cent = []

x1_1 = 2
y1_1 = 4
x2_1 = 66
y2_1 = 67

x1_2 = 3
y1_2 = 5
x2_2 = 68
y2_2 = 69

x1_3 = 6
y1_3 = 7
cent.append([[x1_1,y1_1]])
cent.append([[x2_1,y2_1]])

cent[0].append([x1_2,y1_2])
cent[1].append([x2_2,y2_2])

cent[0].append([x1_3,y1_3])



def stitch_centroids(self, centroids_frame,f):
        
        # make a list of objects tracked in the previous frame
        prev_obj_inds = []
        centroids_prev = []
        j = 0
        for i in range(len(self.centroids_raw)):
            if self.first_frames_raw[i] + len(self.centroids_raw[i]) == f:
                prev_obj_inds.append(i)
                centroids_prev.append(self.centroids_raw[i][-1])
                j += 1
                
        # if no objects were tracked in the previous frame, all objects in the
        # current frame are new objects and no stitching is needed
        if len(centroids_prev) == 0:
            for i in range(len(centroids_frame)):
                self.centroids_raw.append([centroids_frame[i]])
                self.first_frames_raw.append(int(f))
                if self.centerline_method is not None:
                    self.centerlines_raw.append([centerlines_frame[i]])
        
        # if no objects were tracked in the current frame, do nothing
        elif len(centroids_frame) == 0:
            pass
        
        # else create a matrix of the distances between the objects tracked in
        # the previous frame and those tracked in the current frame
        else:
            d_mat = np.empty((len(centroids_prev),len(centroids_frame)))
            for row, cent_prev in enumerate(centroids_prev):            
                for col, cent_frame in enumerate(centroids_frame):
                    d_mat[row,col] = np.linalg.norm(cent_prev-cent_frame)    
            
            # find the closest centroids, second closest, etc., crossing off
            # matched objects until either all worms from the previous frame
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
            # centerlines

            # tracked objects
            tracked_inds = []
            for i in range(len(pair_list)):
                tracked_inds.append(pair_list[i][1])
                self.centroids_raw[prev_obj_inds[pair_list[i][0]]].append([centroids_frame[pair_list[i][1]]])
                if self.centerline_method is not None:
                    self.centerlines_raw[prev_obj_inds[pair_list[i][0]]].append([centerlines_frame[pair_list[i][1]]])
            
            for i in range(len(centroids_frame)):
                if i not in tracked_inds:
                    self.centroids_raw.append([centroids_frame[i]])
                    self.first_frames_raw.append(int(f))
                    if self.centerline_method is not None:
                        self.centerlines_raw.append([centerlines_frame[i]])
