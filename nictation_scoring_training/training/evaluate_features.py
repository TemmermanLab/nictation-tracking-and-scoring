# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:52:14 2022

This script uses a variety of methods to evaluate the utility of various
features for scoring nictation behavior

@author: PDMcClanahan
"""


def evaluate_features(video_file, model_type):
    
    # load features
    
    # load manual scores
    
    # correlation between features
    
    # recursive feature elimination
    
    # effectiveness of each feature by itself (normalized by chance assuming
    # ratios are correct)
    
    # effectiveness of model when one feature at a time is scrambled
    
    
    
    
    
    
    
    
    
    pass





# testing
if __name__ == '__main__':
    try:
        
        # vf = r"C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation" +\
        #     r"\nictation_scoring_training\training\test.avi"
        vf = "C:\\Users\\Temmerman Lab\\Desktop\\Celegans_nictation_dataset"+\
            "\\Ce_R2_d21.avi"
        
        mt = 'random forest'
        
        evaluate_features(vf, mt)
        
    except:
        
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)