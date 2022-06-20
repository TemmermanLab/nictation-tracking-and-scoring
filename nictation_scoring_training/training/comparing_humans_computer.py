# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:23:11 2022

This script compares human nictation scores to each other as well as to the
computer scores

@author: Temmerman Lab
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
#import os

import nict_scoring_functions as nsf
import plotting as nplt



# load the scores from the various humans and plot as heatmaps
Luc_scorefile = r'C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Luc_scoring\manual_nictation_scores.csv'
scores_Luc = nplt.load_scores_csv(Luc_scorefile)
nplt.plot_scores(scores_Luc,'Manual Nictation Scores - Luc', 0, 'luc_scores.png')

Anh_scorefile = r'C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\TuanAnh_scoring\manual_nictation_scores.csv'
scores_Luc = nplt.load_scores_csv(Anh_scorefile)
nplt.plot_scores(scores_Luc,'Manual Nictation Scores - Tuan Anh', 0, 'anh_scores.png')

Pat_scorefile = r'C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset\Ce_R2_d21_tracking\manual_nictation_scores.csv'
scores_Pat = nplt.load_scores_csv(Pat_scorefile)
nplt.plot_scores(scores_Pat,'Manual Nictation Scores - Pat', 0, 'pat_scores.png')


# load the "gold standard" scores


# train a model


# generate and plot automated scores