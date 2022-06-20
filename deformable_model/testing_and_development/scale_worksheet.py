# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:41:22 2021

@author: Temmerman Lab
"""

# pop up a separate cv2 window to draw a line for the scale
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
                cv2.line(imgc, (clicks_x[-2],clicks_y[-2]), (clicks_x[-1],clicks_y[-1]), (0,0,255), 2)
                cv2.circle(imgc,(clicks_x[-1],clicks_y[-1]),2,(0,0,255),-1)
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
            um = simpledialog.askfloat("Input", "Enter the distance in microns:",
                                 parent=dialogue_box,
                                 minvalue=0, maxvalue=1000000)
            
            dialogue_box.destroy()
            dialogue_box.quit()
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


if __name__ == "__main__":
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import simpledialog

    try:
        img = cv2.imread(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\full_frame.png',cv2.IMREAD_GRAYSCALE)
        um_per_pix, scale_img = draw_scale(img)
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)






