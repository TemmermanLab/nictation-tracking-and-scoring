# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:18:09 2021


issues and improvements:
    -display ROI numbers in the GUI window so that ROIs can be removed by
     number more easily
    -add a feature which fits the boundary to a spline instead of line
     for more smoothness
    -cache ROI pixels so that overdrawn ROIs are fully restored when the
    'undo' button is pressed
    -save ROIs already drawn when the 'back' button is pressed before the
    'next' button
    -ability to finish sementing the images in a folder and then continue to
    another folder without closing the GUI
    -automatically restart where you left off in a set of images
    -support multiple object classes
    -show ROIs on the large segmentation GUI
    -keep using new random or other colors after the preset list of colors is
     exhausted.


@author: Temmerman Lab
"""

import tkinter as tk
import tkinter.filedialog as filedialog # necessary to avoid error
from tkinter import simpledialog
import numpy as np
from PIL import Image, ImageTk
import os
import cv2
import copy
from scipy import interpolate
import matplotlib.pyplot as plt


# segmentation GUI
def segmentation_GUI():
    
    # internal functions
    
    # pop up a separate cv2 window to draw a new ROI
    def draw_ROI(img,mask):
        #import pdb; pdb.set_trace()
        fw = 15 # frame width, makes it easier to segment object touching the edge
        img_framed = np.zeros((np.shape(img)[0]+2*fw,np.shape(img)[1]+2*fw),dtype = 'uint8')
        img_framed[fw:-fw,fw:-fw] = img
        imgc = cv2.cvtColor(img_framed,cv2.COLOR_GRAY2RGB)
        img_reset = copy.copy(imgc)
        w_num = np.max(mask)+1
        clicks_x, clicks_y = [],[]
        z = []
        
        def draw_line(event,x,y,flags,param):
            nonlocal imgc, img_reset
            
            if event == cv2.EVENT_MBUTTONDBLCLK:
                print(len(clicks_x))
                if len(clicks_x) > 2:
                    # cv2.line(imgc, (clicks_x[-1],clicks_y[-1]), (clicks_x[0],clicks_y[0]), (0,0,255),1)
                    contours = np.flipud(np.rot90(np.vstack((clicks_x,clicks_y))))
                    cv2.fillPoly(imgc, pts =[contours], color=(0,0,255))
                    z.append(1)
                else:
                    z.append(1)
                imgc = imgc[fw:-fw,fw:-fw,:] # remove frame
            
            elif event == cv2.EVENT_MBUTTONDOWN:
                if len(clicks_x) > 2:
                    cv2.line(imgc, (clicks_x[-1],clicks_y[-1]), (clicks_x[0],clicks_y[0]), (0,0,255),1)
                    contours = np.flipud(np.rot90(np.vstack((clicks_x,clicks_y))))
                    cv2.fillPoly(imgc, pts =[contours], color=(0,0,255))
                    clicks_x.clear(); clicks_y.clear()
                    img_reset = copy.copy(imgc)
                             
            elif event == cv2.EVENT_LBUTTONDOWN and event != cv2.EVENT_LBUTTONDBLCLK:
                clicks_x.append(x)
                clicks_y.append(y)
                print(len(clicks_x))
                if len(clicks_x) ==1:
                    cv2.circle(imgc,(x,y),0,(0,0,255),-1)
                if len(clicks_x) > 1:
                    cv2.line(imgc, (clicks_x[-2],clicks_y[-2]), (clicks_x[-1],clicks_y[-1]), (0,0,255), 1)
             
            elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONDBLCLK:
                if len(clicks_x) >0:
                    clicks_x.pop()
                    clicks_y.pop()
                #imgc = cv2.cvtColor(img_framed,cv2.COLOR_GRAY2RGB)
                imgc = copy.copy(img_reset)
                if len(clicks_x) ==0:
                    pass
                if len(clicks_x) ==1:
                    cv2.circle(imgc,(clicks_x[0],clicks_y[0]),0,(0,0,255),-1)
                if len(clicks_x) > 1:
                    for p in range(len(clicks_x)-1):
                        cv2.line(imgc, (clicks_x[-2-p],clicks_y[-2-p]), (clicks_x[-1-p],clicks_y[-1-p]), (0,0,255), 1)
                 
        cv2.namedWindow('Image to segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Image to segment',imgc)
        cv2.setMouseCallback('Image to segment',draw_line)
        
        while(1):
            cv2.imshow('Image to segment',imgc)
            k = cv2.waitKey(20) & 0xFF
            if len(z)>0:
                print('done')
                cv2.imshow('Image to segment',imgc) # repeating these lines ensure display of pt 2
                k = cv2.waitKey(20) & 0xFF
                cv2.destroyAllWindows()
                print('Points chosen, filling ROI...')
                break
        
        # update mask with new ROI
        mask[np.where((imgc==[0,0,255]).all(axis=-1))] = w_num
        
        return mask
    
    
    # load a new image and the corresponding mask if it exists
    def load_image():
        nonlocal mask, w
        global img
        img = cv2.imread(img_path+'/'+img_list[i],cv2.IMREAD_GRAYSCALE)
        if os.path.isfile(mask_path+'/'+img_list[i][0:-4]+'_mask.png'):
            mask = cv2.imread(mask_path+'/'+img_list[i][0:-4]+'_mask.png',cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(np.shape(img),dtype = 'uint8')
        #if i > 0: import pdb; pdb.set_trace()
        w = np.max(mask)   
    
    # update the main GUI window to show the current image and ROIs already
    # drawn
    def update_disp():
        print('update image function called')
        nonlocal img_win, win_sz, i, i_tot, w, mask
        global disp
        inf_txt['text']='image '+str(i+1) + ' of ' + str(i_tot) + ', '+str(w)+' ROIs'
        disp = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
        for rr in range(w+1):
            if rr < len(mask_colors):
                disp[np.where(mask==rr+1)[0],np.where(mask==rr+1)[1]]= mask_colors[int(rr)]
            else:
                disp[np.where(mask==rr+1)[0],np.where(mask==rr+1)[1]]= mask_colors[-1]
        
        disp = Image.fromarray(disp)
        disp = disp.resize(win_sz,Image.NEAREST) 
        disp = ImageTk.PhotoImage(disp)
        img_win.configure(image = disp)
        img_win.update()
  
    
    def save_mask():
        print('save mask function called')
        # nonlocal scores
        # scores[w][f] = score


    def write_txt_file():
        print('writing text file...')
        txt_file =  os.path.dirname(img_path)+ '/added-object-list.txt'
        lines = []
        lines.append('# image name (\\t) object index\n')
        for img in range(i_tot):
            if os.path.isfile(mask_path+'/'+img_list[i][0:-4]+'_mask.png'):
                msk = cv2.imread(mask_path+'/'+img_list[img][0:-4]+'_mask.png',cv2.IMREAD_GRAYSCALE)
                lines.append(img_list[img]+'\t'+str(np.max(msk))+'\n')
        
        if os.path.isfile(txt_file): os.remove(txt_file)
        
        txt_file_obj = open(txt_file,"w")
        
        # for line in lines:
        #     txt_file_obj.write(line)
        txt_file_obj.writelines(lines)
        txt_file_obj.close()
        
        # write individual text files for each image
        
        # for img in range(i_tot):
        #     txt_file =  os.path.dirname(img_path)+ '/added-object-list.txt'
        #     lines = []
        #     lines.append('# image name \t object index\n')
        #     for img in range(i_tot):
        #         if os.path.isfile(mask_path+'/'+img_list[i][0:-4]+'_mask.png'):
        #             msk = cv2.imread(mask_path+'/'+img_list[img][0:-4]+'_mask.png',cv2.IMREAD_GRAYSCALE)
        #             lines.append(img_list[img]+'\t'+str(np.max(msk))+'\n')
            
        #     if os.path.isfile(txt_file): os.remove(txt_file)
            
        #     txt_file_obj = open(txt_file,"w")
            
        #     # for line in lines:
        #     #     txt_file_obj.write(line)
        #     txt_file_obj.writelines(lines)
        #     txt_file_obj.close()
        
        
        print('...text file written')
        
    
    # button functions
    
    def segment_ROI_button():
        nonlocal w, mask
        print('next worm button pressed')
        mask = draw_ROI(img,mask)
        w = w + 1
        update_disp()

    def undo_by_number_button():
        print('undo by number button pressed')
        nonlocal mask,w
        # dialog to get number to eliminate
        w_elim = simpledialog.askinteger("User Input", "Enter ROI to eliminate:",
                                 parent = segmentation_GUI,
                                 minvalue=1, maxvalue=1000)
        #import pdb; pdb.set_trace()
        if w_elim <= w and w > 0:
            mask[np.where(mask==w_elim)[0],np.where(mask==w_elim)[1]]= 0
            for i in range(w_elim+1,w+1,1):
                mask[np.where(mask==i)[0],np.where(mask==i)[1]]= i-1
            w = w - 1
        update_disp()
            
    def undo_button():
        print('undo button pressed')
        nonlocal mask, w
        mask[np.where(mask==w)[0],np.where(mask==w)[1]]= 0
        if w > 0:
            w = w - 1
        update_disp()
    
    def load_images_button():
        nonlocal img_path, mask_path, img_list, i_tot, i
        global img
        print(img_path)
        if img_path == 'null':
            root = tk.Tk()
            img_path = tk.filedialog.askdirectory(initialdir = '/', \
                title = "Select the folder containing images to be segmented \
                ...")
            root.destroy()
        else:
            print('images already loaded')
        
        print('loading first image in '+img_path)
        img_list = os.listdir(img_path)
        for v in reversed(range(len(img_list))):
            if len(img_list[v])<4 or img_list[v][-4:] != '.png':
                del(img_list[v])
        
        i_tot = len(img_list)
        
        
        mask_path = os.path.dirname(img_path)+ '/masks/'
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        
        load_image()
        update_disp()


    
    def next_image_button():
        nonlocal i
        print('next image button pressed')
        # save mask
        cv2.imwrite(mask_path+'/'+img_list[i][0:-4]+'_mask.png',mask)
        
        # update num_roi_list
        if len(num_roi_list) == i:
            # create new entry (first time segmenting this image)
            num_roi_list.append(w)
        else:
            # set entry to new value (if back had been used)
            num_roi_list[i] = w
        
        if i < i_tot-1:
            i = i + 1
        if i <= i_tot:
            load_image()
            update_disp()
        else:
            print('all images segmented, press exit to save ROIs and quit')
            

    def back_button():
        nonlocal i
        global img
        print('back button pressed, ROIs on current image and previous will be lost')
        if i > 0:
            cv2.imwrite(mask_path+'/'+img_list[i][0:-4]+'_mask.png',mask)
        
            # update num_roi_list
            if len(num_roi_list) == i:
                # create new entry (first time segmenting this image)
                num_roi_list.append(w)
            else:
                # set entry to new value (if back had been used)
                num_roi_list[i] = w
            i = i - 1
            load_image()
            update_disp()
        else:
            print('this is the first image')

               
    def exit_button():
        nonlocal segmentation_GUI
        print('exit button pressed')
        cv2.imwrite(mask_path+'/'+img_list[i][0:-4]+'_mask.png',mask)
        write_txt_file()
        segmentation_GUI.destroy()
        segmentation_GUI.quit()

    
    # initialize variables
    img_list = []
    i = 0 # current image
    i_tot = -1 # total images in folder
    w = 0 # current roi
    num_roi_list = []
    img_path = 'null'
    mask_path = 'null'
    win_sz = (500,500)
    disp = np.zeros((500,500),dtype = 'uint8') # image displayed in GUI window
    mask = copy.copy(disp)
    mask_colors = [[255,0,0],[255,127,0],[255,255,0],[127,255,0],[0,255,0],\
                   [0,255,127],[0,255,255],[0,127,255],[0,0,255],[127,0,255],\
                   [255,0,255],[255,0,127],[255,125,125]]
    
    # set up
    
    # GUI
    segmentation_GUI = tk.Tk()
    segmentation_GUI.title('Segmentation GUI')
    segmentation_GUI.configure(background = "black")
    
    # informational text
    inf_txt = tk.Label(text = 'load a folder containing images to segment')
    inf_txt.grid(row = 0, column = 0, columnspan = 4, padx = 1, pady = 1)

    
    # image window
    ph = ImageTk.PhotoImage(image = Image.fromarray(disp))
    img_win = tk.Label(segmentation_GUI,image = ph, bg = "black", width = 800)
    img_win.grid(row = 1, column = 0, columnspan = 4, padx = 2, pady = 2)
    

    # buttons
    tk.Button(segmentation_GUI, text = "SEGMENT ROI", command = segment_ROI_button, width = 11) .grid(row = 2, column = 0, columnspan = 2, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "UNDO", command = undo_button, width = 11) .grid(row = 2, column = 2, columnspan = 1, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "UNDO BY NUMBER", command = undo_by_number_button, width = 11) .grid(row = 2, column = 3, columnspan = 1, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "LOAD IMAGES", command = load_images_button, width = 11) .grid(row = 3, column = 0, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "NEXT IMAGE", command = next_image_button, width = 11) .grid(row = 3, column = 1, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "BACK", command = back_button,width = 11) .grid(row = 3, column = 2, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    tk.Button(segmentation_GUI, text = "EXIT", command = exit_button,width = 11) .grid(row = 3, column = 3, padx=1, pady=1, sticky = 'W'+'E'+'N'+'S')
    
    
    # # settings for text over images
    # font = cv2.FONT_HERSHEY_SIMPLEX 
    # text_origin = [5, 13]
    # line_spacing = 13
    # text_origin_2 = copy.copy(text_origin); text_origin_2[1] =  text_origin_2[1] + line_spacing
    # text_origin_3 = copy.copy(text_origin); text_origin_3[1] =  text_origin_3[1] + 2*line_spacing
    # font_scale = 0.4
    # text_colors = [(50, 50, 50),(0, 0,255),(255, 0, 0),(0, 0, 0),(255,255,255)]
    # text_thickness = 1
    
    
    
    # run GUI
    
    # # this works to show the image
    # img1_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\dataset\images\ts1_00000.png'
    # img = cv2.imread(img1_file,cv2.IMREAD_GRAYSCALE)
    # disp = Image.fromarray(img)
    # disp = disp.resize(win_sz,Image.NEAREST) 
    # disp = ImageTk.PhotoImage(disp)
    # #import pdb; pdb.set_trace()
    # img_win.configure(image = disp)
    # img_win.update()
    
    segmentation_GUI.mainloop()

if __name__ == '__main__':
    try:
        segmentation_GUI()
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

