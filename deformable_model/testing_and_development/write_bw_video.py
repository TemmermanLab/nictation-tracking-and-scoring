



import os
import sys
sys.path.append(os.path.split(__file__)[0])

import tracker_classes as tracker


try: 
    trkr = tracker.Tracker(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\data\Steinernema_vids_cropped\Sc_All_smell3_V2_ 21-09-17 15-26-15_crop_1_to_300_inc_3.avi')
    # paramters should load automatically # trkr.set_parameters(False,'max_merge',10,1.0,25,[150,300],10,'','None',20)
    trkr.create_BW_video()
    
except:
    import pdb
    import sys
    import traceback
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)