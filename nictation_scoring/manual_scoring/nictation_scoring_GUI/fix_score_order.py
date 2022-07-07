# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:54:03 2022

This is a script to fix previously-scored 

@author: Temmerman Lab
"""

import os
import pickle
import csv
import numpy as np

vignette_path = r'C:\Users\Temmerman Lab\Desktop\Celegans_nictation_dataset'+\
    r'\TuanAnh_scoring\vignettes'

save_file_pickle = os.path.dirname(vignette_path) + \
    r'\manual_nictation_scores_fixed.p'
save_file_csv = os.path.dirname(vignette_path) + \
    r'\manual_nictation_scores_fixed.csv'



# # put in list in natural order (in newer version of nictation_scoring_GUI)
# vignette_list_unsorted = os.listdir(vignette_path)

# digit_list = []
# for v in vignette_list_unsorted:
#     dig = ''
#     for c in v:
#         if c.isdigit():
#             dig += c
#     digit_list.append(int(dig))

# vignette_list = [x for _,x in sorted(zip(digit_list,vignette_list_unsorted))]


# fix existing score sheets

def save_scores_pickle(scores, save_file_pickle):
    pickle.dump(scores, open(save_file_pickle,'wb'))


def save_scores_csv(scores, save_file_csv):
    with open(save_file_csv, mode='w',newline="") as csv_file:

        scores_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL)
        # complicated because csv writer only writes rows
        row = []
        for ww in range(len(scores)): row.append('worm '+str(ww))
        scores_writer.writerow(row)
        
        num_frames = []
        for s in scores: num_frames.append(len(s))
        num_r = np.max(num_frames)
        for r in range(num_r):
            row = []
            for ww in range(len(scores)):
                if r < len(scores[ww]):
                    row.append(scores[ww][r])
                else:
                    row.append('')
            scores_writer.writerow(row)
                

scores_unsorted = pickle.load( open(os.path.dirname(vignette_path)+ \
                                       '/manual_nictation_scores.p','rb'))

vignette_list_unsorted = os.listdir(vignette_path)

digit_list = []
for v in vignette_list_unsorted:
    dig = ''
    for c in v:
        if c.isdigit():
            dig += c
    digit_list.append(int(dig))

scores = [x for _,x in sorted(zip(digit_list,scores_unsorted))]
vignette_list = [x for _,x in sorted(zip(digit_list,vignette_list_unsorted))]

save_scores_pickle(scores, save_file_pickle)
save_scores_csv(scores, save_file_csv)