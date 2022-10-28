# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:14:50 2022

This module contains methods for writing and reading .csv file of tracking
inputs and outputs.

Known issues and potential improvements:


@author: Temmerman Lab
"""

import os
import csv
import numpy as np



def save_params_csv(params, save_path, save_name = 'tracking_parameters'):
    
    if not os.path.exists(save_path):
        print('Creating directory for tracking parameters and output: '+ \
              save_path)
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



def load_parameter_csv(csv_filename):
    parameters = dict()
    
    
    with open(csv_filename, newline="") as csv_file: 
        parameters_reader = csv.reader(csv_file, delimiter=';',
                                       quotechar='"')
        for r in parameters_reader:
            if r[0] == 'Parameter' or r[1] == '':
                pass
            elif r[0] == 'human_checked':
                if r[1]  == 'True' or r[1]  == 'TRUE':
                    r[1] = True
                else:
                    r[1] = False
                
            elif r[0] == 'area_bnds':
                transdict = {91: None, 93: None, 40: None, 41: None,
                             44: None}
                r[1] = r[1].translate(transdict).split(sep=' ')
                r[1] =  [int(r[1][n]) for n in range(len(r[1]))]
            elif r[0] == 'k_sig' or r[0] == 'um_per_pix' or \
                r[0] == 'behavior_sig':
                try:
                    r[1] = float(r[1])
                except:
                    pass
                
            elif r[0] == 'bkgnd_meth' or r[0] == 'mask_RCNN_file' or r[0] == \
                'behavior_model_file': # strings
                pass
            
            else: # all other params should be integers
                r[1] = int(r[1])
                    
            if r[0] != 'Parameter':
                parameters[r[0]] = r[1]
    
    return parameters


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



def load_centroids_csv(centroids_file):
    '''Reads the centroids and first frames in the .csv <centroids_file
    and returns them in the format used within the tracking code'''
    
    # load row by row
    xs = []
    ys = []
    first_frames = []
    with open(centroids_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count == 0:
                print(f'Column names are {", ".join(row)}')
                row_count += 1
            elif np.mod(row_count,2)==0:
                ys.append(np.array(row[1:],dtype='float32'))
                row_count += 1
            else:
                first_frames.append(row.pop(0))
                xs.append(np.array(row,dtype='float32')); row_count += 1
    
    # reshape into the proper format
    centroids = []
    for w in range(len(xs)):
        centroids_w = []
        for f in range(len(xs[w])):
            centroids_w.append(np.array((xs[w][f],ys[w][f])))
        centroids.append(centroids_w)
    
    first_frames = [int(ff) for ff in first_frames]
    
    return centroids, first_frames



def save_centerlines(centerlines, centerline_flags, first_frames, 
                     save_path, save_dir = 'centerlines'):

    if not os.path.exists(save_path + '\\' + save_dir):
        print('Creating directory for centerlines csvs and other output: '
              +save_path+'\\' + save_dir)
        os.makedirs(save_path+'\\' + save_dir)
    
    for w in range(len(centerlines)):
        save_file_csv = save_path + '\\' + save_dir + '\\' + \
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
                
            
    print("Centerlines saved in " + save_path + '\\' + save_dir)
    


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



def load_centerlines_csv(save_path, N = 50):
    '''Loads the centerlines saved as .csv files in <save_path> and 
    returns them as well as their flags'''
    
    centerlines = list()
    centerline_flags = list()
    centerline_files = os.listdir(save_path)
    w = 0
    for file in range(len(centerline_files)):
        csv_filename = save_path + '\\' + centerline_files[file]
        if csv_filename[-15:] == 'worm_' + "{:06d}".format(w) + '.csv':
            with open(csv_filename, newline="") as csv_file: 
                centerlines_reader = csv.reader(csv_file, delimiter=',',
                                                quotechar='"')
                for r in centerlines_reader:
                    if r[0]=='frame':
                        numf = 1+int(r[-1])-int(r[1]); f = 0
                        centerlines_worm = np.empty((numf,1,N,2))
                        
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
        centerlines.append(list(centerlines_worm))
        centerline_flags.append(centerline_flags_worm)
        
    return centerlines, centerline_flags