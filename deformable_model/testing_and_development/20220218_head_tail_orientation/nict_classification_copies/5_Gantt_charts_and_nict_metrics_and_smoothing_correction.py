# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:51:07 2021

This script uses the nictation indicators and their first derivatives, scaled
to a Gaussian, to train a Random Forest model.  This model is used to predict
becavior based on the indicators in a portion of the data reserved for testing.
The results are displayed as a Gantt chart, showing the types of errors.

Nictation metrics (Lee et al 2012) are calculated from the manual and machine-
scored behavior and compared.

An attempt is made to address the problem of
isolated misclassified frames by smoothing the probabilities.

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


# dataframe with NaNs removed and scaled
df_masked = models.nan_mask_dataframe(dataframe)
df_scaled, scaler = models.scale_data(df_masked,method = 'Gaussian')

# Gantt charts of behavior
# manually-scored worm 55:82
plotting.plot_scores(behavior_scores[55:], title = 'Manual Nictation Scores',
                     figsize = (6,12),w_tick_start = 54)
plotting.plot_scores(behavior_scores, title = 'Manual Nictation Scores')

# dataframe is 50271 rows
f_train = 0
for w in range(55): f_train += len(behavior_scores[w])
# first 38418 are for training (0.764)
df_scaled = copy.deepcopy(dataframe)
df_scaled = models.nan_mask_dataframe(df_scaled)
X_train_spl, X_test_spl, y_train_spl, y_test_spl = models.split(df_scaled,prop_train = 0.764)
model, train_acc, test_acc, probabilities, predictions = \
    models.learn_and_predict(X_train_spl, X_test_spl, y_train_spl, 
    y_test_spl,'random forest')
predictions2 = model.predict(X_test_spl)

# get predictions from the original scores
# model_scores = copy.copy(scores)
# model_scores[:] = -2
model_scores = []
for w in range(np.shape(metric_scores)[1]):
    w_df = models.worm_to_df(w,metric_labels,behavior_scores[w],activity[w],metric_scores)
    w_df = models.nan_mask_dataframe(w_df)
    #w_df = models.scale_min_max(w_df, 'min max')
    X = w_df[w_df.columns[2:]]
    model_predictions = model.predict(X) 
    # model_scores[w,0:len(model_predictions)]=model_predictions
    model_scores.append(model_predictions)

plotting.plot_scores(model_scores[55:],'Random Forest Nictation Scores',figsize=(6,12),w_tick_start = 54)


# nictation metrics
import CNI_nictation_metrics as nict

ratios_man = []
ratios_comp = []
for w in range(55,len(model_scores)):
    ratios_man.append(nict.nictation_ratio(behavior_scores[w], only_active = True))
    ratios_comp.append(nict.nictation_ratio(model_scores[w], only_active = True))

rng_man = np.linspace(-.3,.3,num = len(ratios_man))
rng_comp = np.linspace(-.3,.3,num = len(ratios_comp))

for w in reversed(range(len(ratios_man))):
    # if np.isnan(ratios_man[w]) or np.isinf(ratios_man[w]):
    #     ratios_man.pop(w)
    if np.isinf(ratios_man[w]):
        ratios_man[w] = np.nan

rng_man = np.linspace(-.3,.3,num = len(ratios_man))
rng_comp = np.linspace(-.3,.3,num = len(ratios_comp))+1

fig,axs = plt.subplots(figsize = [3,4])
plt.scatter(rng_man,np.array(ratios_man),marker = '.',c = 'k',s=150)
plt.scatter(rng_comp,np.array(ratios_comp),marker = '.',c = 'k',s=150)
axs.set_xticks([0,1])
axs.set_xticklabels(['manual','computer'])
axs.set_ylabel('nictation ratio')
plt.title('Nictation Ratio of Worm Traces')
for w in range(len(ratios_man)):
    if ~np.isnan(ratios_man[w]):
        plt.plot([rng_man[w],rng_comp[w]],[ratios_man[w],ratios_comp[w]],c='k')

ratio_man = nict.nictation_ratio(behavior_scores[55:], only_active = True)
ratio_comp = nict.nictation_ratio(model_scores[55:], only_active = True)
plt.scatter(0,ratio_man,c = 'r',s=300,marker = '+')
plt.scatter(1,ratio_comp,c = 'r',s=300,marker = '+')
plt.show()



init_r_man = nict.initiation_rate(behavior_scores[55:])
init_r_comp = nict.initiation_rate(model_scores[55:])



duration_man,episodes_man = nict.nictation_duration(behavior_scores[55:],exclude_partial_episodes = True)
duration_comp,episodes_comp = nict.nictation_duration(model_scores[55:],exclude_partial_episodes = True)

    
rng_man = np.linspace(-.3,.3,num = len(episodes_man))
rng_comp = np.linspace(-.3,.3,num = len(episodes_comp))+1

for w in reversed(range(len(episodes_man))):
    # if np.isnan(ratios_man[w]) or np.isinf(ratios_man[w]):
    #     ratios_man.pop(w)
    if np.isinf(ratios_man[w]):
        ratios_man[w] = np.nan


fig,axs = plt.subplots(figsize = [3,4])
plt.scatter(rng_man,np.array(episodes_man),marker = '.',c = 'k',s=150)
plt.scatter(rng_comp,np.array(episodes_comp),marker = '.',c = 'k',s=150)
axs.set_xticks([0,1])
axs.set_xticklabels(['manual','computer'])
axs.set_ylabel('duration (s)')
plt.title('Nictation Episode Duration')
for w in range(len(ratios_man)):
    if ~np.isnan(ratios_man[w]):
        plt.plot([rng_man[w],rng_comp[w]],[ratios_man[w],ratios_comp[w]],c='k')

# ratio_man = nict.nictation_ratio(behavior_scores[55:], only_active = True)
# ratio_comp = nict.nictation_ratio(model_scores[55:], only_active = True)
plt.scatter(0,duration_man,c = 'r',s=300,marker = '+')
plt.scatter(1,duration_comp,c = 'r',s=300,marker = '+')
plt.show()



# try to replicate nictation measures by smoothing the probabilities,
# recalculating the classes, and recalculating the params

df = df_masked


# split into training and test sets at worm 55
X_train, X_test, y_train, y_test = models.split(df,prop_train = 0.764)



# random forest classification
model, train_acc, test_acc, probabilities, predictions = \
    models.learn_and_predict(X_train, X_test, y_train, 
    y_test,'random forest')
predictions2 = model.predict(X_test)
model_scores = []; model_probs = []
for w in range(np.shape(metric_scores)[1]):
    w_df = models.worm_to_df(w,metric_labels,behavior_scores[w],activity[w],metric_scores)
    w_df = models.nan_mask_dataframe(w_df)
    X = w_df[w_df.columns[2:]]
    model_scores.append(model.predict(X))
    model_probs.append(model.predict_proba(X))


# calculate nictation params from manual scores for worms 55+
init_r_man = nict.initiation_rate(behavior_scores[55:])
nict_r_man = nict.nictation_ratio(behavior_scores[55:], only_active = True)
nict_d_man, ep_man = nict.nictation_duration(behavior_scores[55:],exclude_partial_episodes = True)
    
# calculate nictation params w/o smoothing
init_r_auto_raw = nict.initiation_rate(model_scores[55:])
nict_r_auto_raw = nict.nictation_ratio(model_scores[55:], only_active = True)
nict_d_auto_raw, ep_man = nict.nictation_duration(model_scores[55:],exclude_partial_episodes = True)

# calculate nictation params w/ different amounts of smoothing
sigmas = np.linspace(0,3,31)
fps = 5.0
init_rs_smooth = np.empty((len(sigmas)))
nict_rs_smooth = np.empty((len(sigmas)))
nict_ds_smooth = np.empty((len(sigmas)))
probs = model_probs[55:]
for sigma in range(len(sigmas)):
    probs_smooth = models.smooth_probabilities(probs, sigmas[sigma], fps)
    preds = models.probabilities_to_predictions(probs_smooth)
    init_rs_smooth[sigma] = nict.initiation_rate(preds)
    nict_rs_smooth[sigma] = nict.nictation_ratio(preds, only_active = True)
    nict_ds_smooth[sigma],durs = nict.nictation_duration(preds,exclude_partial_episodes = True)
    
# plot manual nictation param, auto params w/o smoothing, and auto params with
# different amounts of smoothing
fig, axes = plt.subplots(figsize=(5,4))
axes.set_xlabel('sigma (s)')
axes.set_ylabel('nictation ratio')
axes.set_title('Effect of Smoothing on Nictation Ratio')
axes.plot(sigmas,nict_rs_smooth, 'ko',markersize = 4,label = 'auto score')
axes.plot(sigmas,nict_r_man*np.ones(np.shape(sigmas)),linestyle = '-',color = 'k',label = 'manual score')
plt.ylim((0,.5))
plt.legend()
plt.show()

fig, axes = plt.subplots(figsize=(5,4))
axes.set_xlabel('sigma (s)')
axes.set_ylabel('initiation rate (Hz)')
axes.set_title('Effect of Smoothing on Initiation Rate')
axes.plot(sigmas,init_rs_smooth, 'ko',markersize = 4,label = 'auto score')
axes.plot(sigmas,init_r_man*np.ones(np.shape(sigmas)),linestyle = '-',color = 'k',label = 'manual score')
plt.ylim((0,.01))
plt.legend()
plt.show()

fig, axes = plt.subplots(figsize=(5,4))
axes.set_xlabel('sigma (s)')
axes.set_ylabel('nictation duration')
axes.set_title('Effect of Smoothing on Nictation Duration')
axes.plot(sigmas,nict_ds_smooth, 'ko',markersize = 4,label = 'auto score')
axes.plot(sigmas,nict_d_man*np.ones(np.shape(sigmas)),linestyle = '-',color = 'k',label = 'manual score')
plt.ylim((0,15))
plt.legend()
plt.show()

sigma = 0.5
probs_smooth = models.smooth_probabilities(probs, sigma, fps)
preds = models.probabilities_to_predictions(probs_smooth)
plotting.plot_scores(preds, title = 'Auto Nictation Scores, Sigma = 0.5 s',figsize = (5.3,5.3),w_tick_start = 55)

