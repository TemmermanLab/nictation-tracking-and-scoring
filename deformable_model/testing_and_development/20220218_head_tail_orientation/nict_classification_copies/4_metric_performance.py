# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:43:11 2021 (with calculate_nictation_metrics)




@author: PDMcClanahan
"""



# USE THESE INDICATORS FOR CLASSIFICATIONS


# import modules
import pickle
import numpy as np
import sys
import copy
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn')
import models

metric_labels = ['ends_mov_bias','out_of_track_centerline_mov',
           'blur','total_curvature','lateral_movement','longitudinal_movement','centroid_path',
           'angular_sweep','body_length','centroid_progress',
           'centroid mvmnt ecc','centroid mvmnt area']


# load data and indicators
behavior_scores = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\051821_quaternary_scoring\manual_nictation_scores.p','rb')) 
metric_scores = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\051821_quaternary_scoring\20210712_custom_metric_scores.p','rb'))  
activity = pickle.load(open(r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\tracking\auto_scoring\for_Ahn\20210212_Cu_ring_test(copy)\dauers 14-14-56_tracking\20210511_diff_img_activity.p','rb'))  


# linearize behavior and activity
fps = 5
act_primed, act_trunc = models.first_derivative_act(activity,fps)


behavior_scores_l = []
for i in range(len(behavior_scores)):behavior_scores_l = behavior_scores_l+list(behavior_scores[i][1:])

# numerize behavior scores
behavior = copy.copy(behavior_scores_l)
for i in range(len(behavior)):
    if behavior_scores_l[i] == 0: behavior[i] = 'quiescent'
    if behavior_scores_l[i] == 1: behavior[i] = 'cruising'
    if behavior_scores_l[i] == 2: behavior[i] = 'waving'
    if behavior_scores_l[i] == 3: behavior[i] = 'standing'
    if behavior_scores_l[i] == -1: behavior[i] = 'censored'

act_l = []
for i in range(len(act_trunc)):
    if i < 6: # first 6 worms tracked from beginning, so there is no activity for frame 1
        act_l = np.concatenate((act_l,act_trunc[i]))
    else:
        act_l = np.concatenate((act_l,act_trunc[i][1:]))
        
act_l_primed = []
for i in range(len(act_primed)):
    if i < 6: # first 6 worms tracked from beginning, so there is no activity for frame 1
        act_l_primed = np.concatenate((act_l_primed,act_primed[i]))
    else:
        act_l_primed = np.concatenate((act_l_primed,act_primed[i][1:]))


# calculate the first derivative of the metric scores
fps = 5
met_primed, met_trunc = models.first_derivative_met(metric_scores,fps)


# create dataframe
met_trunc_ll = []
for i in range(len(met_trunc)):
    met_trunc_l = []
    for j in range(len(met_trunc[i])):
        met_trunc_l = met_trunc_l + list(met_trunc[i][j][1:])
    met_trunc_ll.append(copy.copy(met_trunc_l))

met_primed_ll = []
for i in range(len(met_primed)):
    met_primed_l = []
    for j in range(len(met_primed[i])):
        met_primed_l = met_primed_l + list(met_primed[i][j][1:])
    met_primed_ll.append(copy.copy(met_primed_l))


# concatenate the dataframes
metric_labels_primed = copy.copy(metric_labels)
for l in range(len( metric_labels_primed)): metric_labels_primed[l] = metric_labels_primed[l] + '_primed'


data_primed = {'behavior label': behavior_scores_l,'behavior':behavior, 
        'activity': act_l, 'activity_primed': act_l_primed,
        metric_labels[1]:met_trunc_ll[1],
        metric_labels[2]:met_trunc_ll[2],
        metric_labels[3]:met_trunc_ll[3],
        metric_labels[4]:met_trunc_ll[4],
        metric_labels[5]:met_trunc_ll[5],
        metric_labels[6]:met_trunc_ll[6],
        metric_labels[7]:met_trunc_ll[7],
        metric_labels[8]:met_trunc_ll[8],
        metric_labels[9]:met_trunc_ll[9],
        metric_labels[10]:met_trunc_ll[10],
        metric_labels[11]:met_trunc_ll[11],
        metric_labels_primed[1]:met_primed_ll[1],
        metric_labels_primed[2]:met_primed_ll[2],
        metric_labels_primed[3]:met_primed_ll[3],
        metric_labels_primed[4]:met_primed_ll[4],
        metric_labels_primed[5]:met_primed_ll[5],
        metric_labels_primed[6]:met_primed_ll[6],
        metric_labels_primed[7]:met_primed_ll[7],
        metric_labels_primed[8]:met_primed_ll[8],
        metric_labels_primed[9]:met_primed_ll[9],
        metric_labels_primed[10]:met_primed_ll[10],
        metric_labels_primed[11]:met_primed_ll[11],
        }

dataframe_primed = pd.DataFrame(data=data_primed)
df_masked_primed = models.nan_mask_dataframe(dataframe_primed)
df_masked_primed_scaled,scaler = models.scale_data(df_masked_primed,'Gaussian')


data = {'behavior label': behavior_scores_l,'behavior':behavior, 
        'activity': act_l,
        metric_labels[1]:met_trunc_ll[1],
        metric_labels[2]:met_trunc_ll[2],
        metric_labels[3]:met_trunc_ll[3],
        metric_labels[4]:met_trunc_ll[4],
        metric_labels[5]:met_trunc_ll[5],
        metric_labels[6]:met_trunc_ll[6],
        metric_labels[7]:met_trunc_ll[7],
        metric_labels[8]:met_trunc_ll[8],
        metric_labels[9]:met_trunc_ll[9],
        metric_labels[10]:met_trunc_ll[10],
        metric_labels[11]:met_trunc_ll[11],
        }

dataframe = pd.DataFrame(data=data)
df_masked = models.nan_mask_dataframe(dataframe)


# # histogram of new metrics
# palette = np.array([[  255,   255,   255],   # white - no score
#                     [125,   125,   125],   # gray - censored 
#                     [  0, 0, 80],   # dark blue - quiescent
#                     [  0, 0, 255],   # blue - crawling
#                     [  255,   0, 0],   # red - nictating
#                     [80, 0, 0]])  # dark red - standing
    
# hist_colors = palette[[3,4,5,2,1]]
# behaviors = dataframe.behavior.unique()
# metrics = dataframe.columns[2:]
# behavior_list = dataframe.behavior
# units = ['ratio','microns^2','microns','a.u.','radians',
#          'microns','microns','radians','microns','microns']

# fig, axs = plt.subplots(1,2,figsize=(4,2),tight_layout=True)
# indr = 0
# indc = 0
# ind = 0
# for m in metrics[-2:]:
#     metric_list = np.array(dataframe[m])
#     cats = []
#     for b in behaviors[0:2]:
#         cats.append(metric_list[np.where(behavior_list==b)])
    
#     e0 = np.min((np.nanmin(cats[0]),np.nanmin(cats[1])))
#     ee = np.max((np.nanmax(cats[0]),np.nanmax(cats[1])))
#     bin_edges = np.linspace(e0,ee,16) 
    
#     for b in range(len(behaviors[0:2])):
#         axs[indc].hist(cats[b], bins=bin_edges, alpha=0.5, label=behaviors[b],
#                  density = True, color = hist_colors[b]/255)
#     #axs[ind].xlabel("Data", size=14)
#     #axs[ind].ylabel("Count", size=14)
#     axs[indc].title.set_text(m)
#     axs[indc].set(xlabel=units[ind])
#     if indc in [0,1,2,3]:
#         indc = indc+1
#     else:
#         indc = 0
#         indr = indr+1
#     ind = ind+1
# plt.show()

# # scatter of new metrics
# metrics = dataframe.columns[2:]
# behaviors = dataframe.behavior.unique()
# behavior_list = dataframe.behavior
# sp_colors = palette[[3,4,5,2,1]]
# fig, axs = plt.subplots(1,1,figsize=(3,3),tight_layout=True)
# plt.title('Scattermatrix of Centroid Mvmnt Area and Eccentricity')
# plt.xlabel('centroid mvmnt ecc (ratio)')
# plt.ylabel('centroid mvmnt area (microns^2)')


# indr = 0
# indc = 0
# ind = 0
# # for m1 in np.arange(len(metrics)-1):
# #     for m2 in np.arange(1,len(metrics)):
# m1 = 10; m2 = 11
# # if m2-1-m1 >= 0:
# metric_list_1 = np.array(dataframe[metrics[m1]])
# metric_list_2 = np.array(dataframe[metrics[m2]])
# cats_1 = []
# cats_2 = []
# for b in behaviors:
#     cats_1.append(metric_list_1[np.where(behavior_list==b)])
#     cats_2.append(metric_list_2[np.where(behavior_list==b)])

# for b in range(len(behaviors)):
#     axs.scatter(cats_1[b],cats_2[b], color = sp_colors[b]/255,
#             alpha=0.3, marker = '.',s=1)
#     if m1 == 0:
#         axs.set(ylabel=metrics[m2])
#     if m2-1 == 8:
#         axs.set(xlabel=metrics[m1])
# # else:
# #     axs[m2-1,m1].axis('off')

# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], color=palette[3]/255, lw=10),
#                 Line2D([0], [0], color=palette[4]/255, lw=10),
#                 Line2D([0], [0], color=palette[2]/255, lw=10),
#                 Line2D([0], [0], color=palette[5]/255, lw=10),
#                 Line2D([0], [0], color=palette[1]/255, lw=10)]
# labels = ['cruising','waving','quiescent','standing',
#           'censored']

# fig.legend(custom_lines, labels,loc = 'right')
# plt.show()



# scale with Gaussian, replot
model_types = ['logistic regression','decision tree','k nearest neighbors',
          'linear discriminant analysis','Gaussian naive Bayes',
          'support vector machine','random forest','neural network']
X_train, X_test, y_train, y_test = models.split(df_masked_primed_scaled)


train_scores = []
test_scores = []

heatmap1 = np.empty((len(model_types),2))

for mtype in range(len( model_types)):

    
    model, train_acc, test_acc, probabilities, predictions = \
    models.learn_and_predict(X_train, X_test, y_train, 
    y_test,model_types[mtype])
    heatmap1[mtype,0:2] = [train_acc,test_acc]


# plot heatmap
fig, axes = plt.subplots()
im = axes.imshow(heatmap1,cmap='RdBu')
plt.title('Model Performance')
axes.xaxis.set_label_position('top')
axes.xaxis.tick_top() 
axes.set_xticks([0,1])
axes.set_xticklabels(['train','test'])
axes.set_yticks(np.arange(len(model_types)))
axes.set_yticklabels(model_types)
# axes.set_xlabel('')
plt.setp(axes.get_xticklabels(),rotation = 0, ha = 'center', rotation_mode = 'anchor')
fw = ['normal','bold']
for i in range(2):
    for j in range(len(model_types)):
        text = axes.text(i,j,"%0.2f" % heatmap1[j,i],ha='center',va='center',fontweight = fw[i])
plt.show()



# decrease in performance of random forest model with each metric scrambled and
# increase with all but one metric scrambled (vs all scrambled)

# make the model
model, train_acc, test_acc, probabilities, predictions = \
    models.learn_and_predict(X_train, X_test, y_train, 
    y_test,'random forest')


# performance on test set
test_perf = model.score(X_test,y_test) # same as test_acc above

# performance on test set with one scrambled
perfs_removed = []
for col in X_test.columns:
    X_test_scram = copy.deepcopy(X_test)
    scram_col = np.array(X_test_scram[col])
    np.random.shuffle(scram_col)
    X_test_scram[col] = scram_col
    perfs_removed.append(model.score(X_test_scram,y_test))
decrements = perfs_removed-test_perf

# performance on a test set with all first derivitives scrambled
X_test_scram = copy.deepcopy(X_test)
for col in X_test.columns[13:]:
    scram_col = np.array(X_test_scram[col])
    np.random.shuffle(scram_col)
    X_test_scram[col] = scram_col
perf_deriv_scram = model.score(X_test_scram,y_test)

# performance on totally-scrambled test set
X_test_scram = copy.deepcopy(X_test)
for col in X_test.columns:
    scram_col = np.array(X_test_scram[col])
    np.random.shuffle(scram_col)
    X_test_scram[col] = scram_col
perf_all_scram = model.score(X_test_scram,y_test)

# performance on test set with one metric unscrambled
perfs_added = []
for col in X_test.columns:
    X_test_unscram = copy.deepcopy(X_test_scram)
    X_test_unscram[col] = X_test[col]
    perfs_added.append(model.score(X_test_unscram,y_test))

increments = perfs_added-perf_all_scram

# plot indicator performance heatmap
heatmap2 = np.vstack((decrements, increments))
fig, axes = plt.subplots()
im = axes.imshow(heatmap2,cmap='RdBu')
plt.title('Contribution of Metrics to Random Forest Performance')
axes.xaxis.set_label_position('bottom')
axes.set_xticks(np.arange(len(X_test.columns)))
axes.set_xticklabels(X_test.columns,rotation = 45, ha="right")
axes.set_yticks([0,1])
axes.set_yticklabels(['decrement','increment'])
fw = ['normal','normal']
for i in range(2):
    for j in range(len(X_test.columns)):
        text = axes.text(j,i,"%0.2f" % heatmap2[i,j],ha='center',va='center',fontweight = fw[i],fontsize=5)
plt.show()


heatmap3 = np.vstack((decrements[0:13], increments[0:13]))
fig, axes = plt.subplots()
im = axes.imshow(heatmap3,cmap='RdBu')
plt.title('Contribution of Metrics to Random Forest Performance')
axes.xaxis.set_label_position('bottom')
axes.set_xticks(np.arange(len(X_test.columns[0:13])))
axes.set_xticklabels(X_test.columns[0:13],rotation = 45, ha="right")
axes.set_yticks([0,1])
axes.set_yticklabels(['decrement','increment'])
fw = ['bold','bold']
for i in range(2):
    for j in range(len(X_test.columns[0:13])):
        if i == 0:
            text = axes.text(j,i,"%0.2f" % abs(heatmap3[i,j]),ha='center',va='center',fontweight = fw[i],fontsize=10,color = [0,0,0],rotation = 45)
        else:
            text = axes.text(j,i,"%0.2f" % abs(heatmap3[i,j]),ha='center',va='center',fontweight = fw[i],fontsize=10,color = [0,0,0],rotation = 45)
plt.show()


plt.scatter(increments,abs(decrements),color = 'k')
plt.title('Increment vs. Decrement by Indicator')
plt.xlabel('increment')
plt.ylabel('decrement')
plt.show

# Gantt charts of behavior





