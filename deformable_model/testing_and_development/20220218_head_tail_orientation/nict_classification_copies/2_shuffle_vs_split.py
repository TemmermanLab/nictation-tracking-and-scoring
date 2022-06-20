# -*- coding: utf-8 -*-
"""
Modified from 20210604_figs_for_nictation_mtg

This script uses nictation indicators and manual scores to train several types
of ML model from SciKit learn to classify dauer behavior as censored, cruising,
waving, or standing. It shows that shuffling the training data results in 
apparently better performance. However, this is likely because the model is
learning on examples taken from long bouts of similar behavior by individual
worms. In real use, this will not be the case, so splitting the training data
(ie testing on completely different tracks) is more indicative of real-world
use. However, one caveat is that even though the data is split, the same
individual animals could appear in the training and test set because (1) one
track is plit in two at the split point and (2) tracking failures mean that
multiple tracks are often of the same animal.

@author: PDMcClanahan
"""

import pickle
import numpy as np
import sys
import copy
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn')
import models as models


# load data
# original files in E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\051821_quaternary_scoring
behavior_scores = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\051821_quaternary_scoring\manual_nictation_scores.p','rb')) 
metric_scores = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20211128_indicator_vals.p','rb'))  
activity = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\20210511_diff_img_activity.p','rb'))  



behavior_scores_l = []
for i in range(len(behavior_scores)):behavior_scores_l = behavior_scores_l+list(behavior_scores[i])

activity_l = []
for i in range(len(activity)):
    if i < 6: # first 6 worms tracked from beginning, so there is no activity for frame 1
        activity_l = activity_l+activity[i]
    else:
        activity_l = activity_l+activity[i][1:]

metric_scores_ll = []
for i in range(len(metric_scores)):
    metric_scores_l = []
    for j in range(len(metric_scores[i])):
        metric_scores_l = metric_scores_l + list(metric_scores[i][j][1:])
    metric_scores_ll.append(copy.copy(metric_scores_l))

# # older scores (pre 5-24)
# metric_labels = ['end to end mvmnt bias','head/tail mvmnt bias',
#                  'out of trk cline mvmnt','blur','total curvature',
#                  'lateral movement bias','head path','angular sweep','body length',
#                  'centroid progress']



# put in dataframe
metric_labels = ['end to end mov_bias','head/tail mvmnt bias','out of trk cline mvmnt',
           'blur','total curvature','lateral mvmnt bias','centroid path',
           'angular sweep','body length','centroid progress']

behavior = copy.copy(behavior_scores_l)
for i in range(len(behavior)):
    if behavior_scores_l[i] == 0: behavior[i] = 'quiescent'
    if behavior_scores_l[i] == 1: behavior[i] = 'cruising'
    if behavior_scores_l[i] == 2: behavior[i] = 'waving'
    if behavior_scores_l[i] == 3: behavior[i] = 'standing'
    if behavior_scores_l[i] == -1: behavior[i] = 'censored'


data = {'behavior label': behavior_scores_l,'behavior':behavior, 'activity': activity_l,
        metric_labels[1]:metric_scores_ll[1],
        metric_labels[2]:metric_scores_ll[2],
        metric_labels[3]:metric_scores_ll[3],
        metric_labels[4]:metric_scores_ll[4],
        metric_labels[5]:metric_scores_ll[5],
        metric_labels[6]:metric_scores_ll[6],
        metric_labels[7]:metric_scores_ll[7],
        metric_labels[8]:metric_scores_ll[8],
        metric_labels[9]:metric_scores_ll[9],
        }

dataframe = pd.DataFrame(data=data)

#custom metric plots
import CNI_plotting as plotting
plotting.bar_count(dataframe)
plotting.metric_histogram(dataframe)
plotting.scattermatrix(dataframe)


# dataframe with NaNs removed
df_masked = models.nan_mask_dataframe(dataframe)
# pickle.dump(df_masked,open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\051821_quaternary_scoring\df_masked.p','wb'))

# scale dataframe
# df_scaled, scaler = models.scale_min_max(df_masked, 'min max')
# df_scaled, scaler = models.scale_min_max(df_masked, 'variance')
# df_scaled, scaler = models.scale_min_max(df_masked, 'Gaussian')
# df_scaled, scaler = models.scale_min_max(df_masked, 'whiten')


# shuffle data
X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = models.shuffle(df_masked)


# split data
X_train_spl, X_test_spl, y_train_spl, y_test_spl = models.split(df_masked)



# heatmap of model performance on unscaled data, either shuffled or split
X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = models.shuffle(df_masked)
X_train_spl, X_test_spl, y_train_spl, y_test_spl = models.split(df_masked)

model_types = ['logistic regression','decision tree','k nearest neighbors',
          'linear discriminant analysis','Gaussian naive Bayes',
          'support vector machine','random forest']

heatmap = np.empty((len(model_types),4))

for mtype in range(len(model_types)):
    model, train_acc, test_acc, probabilities, predictions = \
        models.learn_and_predict(X_train_shuf, X_test_shuf, y_train_shuf, 
        y_test_shuf,model_types[mtype])
    heatmap[mtype,0:2] = [train_acc,test_acc]
    
for mtype in range(len(model_types)):
    model, train_acc, test_acc, probabilities, predictions = \
        models.learn_and_predict(X_train_spl, X_test_spl, y_train_spl, 
        y_test_spl,model_types[mtype])
    heatmap[mtype,2:4] = [train_acc,test_acc]

fig, axes = plt.subplots()
im = axes.imshow(heatmap,cmap='viridis')
plt.title('Model Performance')
axes.xaxis.set_label_position('top')
axes.xaxis.tick_top() 
axes.set_xticks([0,1,2,3])
axes.set_xticklabels(['train','test','train','test'])
axes.set_yticks(np.arange(len(model_types)))
axes.set_yticklabels(model_types)
axes.set_xlabel('shuffled         split')
plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center', rotation_mode = 'anchor')

for i in range(4):
    for j in range(len(model_types)):
        text = axes.text(i,j,"%0.2f" % heatmap[j,i],ha='center',va='center',fontweight = 'bold')
plt.show()
