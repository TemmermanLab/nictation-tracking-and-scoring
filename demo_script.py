# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:27:13 2022

This script is an example of how to run the tracking code using a script
instead of using the GUI.  In this example all the parameters and settings are
specified in the script for the sake of consistency.

@author: Temmerman Lab
"""

import os
import sys
sys.path.append(os.path.split(__file__)[0])
import tracker as tracker

def main():
    
    
    # designate segmentation method
    segmentation_method = 'intensity'
    # segmentation_method = 'mask_RCNN'
    
    # designate videos to track
    vid_files = []
    vid_files.append(r'C:\Users\Temmerman Lab\Desktop\test_data_for_tracking\R1d4_first_four.avi')
    
    
    # create tracker objects using the specifications above
    trackers = []
    for v in vid_files:
        trackers.append(tracker.Tracker(v, segmentation_method))
        
        
        # designate metaparameters / tracking settings
        trackers[-1].metaparameters['stitching_method'] = 'distance'
        #trackers[-1].metaparameters['stitching_method'] = 'overlap'
        
        trackers[-1].metaparameters['rectification_method'] = 'distance'
        #trackers[-1].metaparameters['rectification_method'] = 'L1_norm'     
        
        
        # set tracking parameters
        trackers[-1].parameters['bkgnd_meth'] = 'max_merge'
        trackers[-1].parameters['bkgnd_nframes'] = 2 
        trackers[-1].parameters['k_sig'] = 1.5
        trackers[-1].parameters['bw_thr'] = 10
        # trackers[-1].parameters['bw_thr'] = 50
        trackers[-1].parameters['area_bnds'] = (200,1500)
        trackers[-1].parameters['d_thr'] = 10
        trackers[-1].parameters['del_sz_thr'] = -1
        trackers[-1].parameters['um_per_pix'] = 4.412
        trackers[-1].parameters['min_f'] = 2
        
    
    # run tracking
    for t in trackers:
        t.track()
        
        
    
if __name__ == '__main__':
    
    try:
        main()
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
