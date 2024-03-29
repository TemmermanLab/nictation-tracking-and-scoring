# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:40:14 2021


This file contains functions for plotting helpful visuals of manual and auto-
matic behavioral scores, custom metrics, etc.


@author: PDMcClanahan
"""


import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


testing = False

palette = np.array([[  255,   255,   255],   # white - no score
                    [125,   125,   125],   # gray - censored 
                    [  0, 0, 80],   # dark blue - quiescent
                    [  0, 0, 255],   # blue - cruising
                    [  255,   0, 0],   # red - waving
                    [80, 0, 0]])  # dark red - standing


def interleaved_scores_plot(scores, to_show, labels):
    '''Plots scores from worm tracks in <to_show> in list of lists <scores>,
    using list of strings <labels> for a key'''
    
    
    
    
    
    plt.show()

# plots a heat map / gantt chart of dauer behavioral scores
def plot_scores(scores, title = 'Manual Nictation Scores',figsize = (5.3,5.3),w_tick_start = 0):
    #import pdb; pdb.set_trace()
    if type(scores)==list:
        scores = np.flip(scores)
        scores_list = copy.deepcopy(scores)
        lens = []
        for w in scores: lens.append(len(w))
        scores = np.zeros((len(scores),np.max(lens)))-2
        for w in range(len(scores_list)):
            scores[w,0:len(scores_list[w])]=scores_list[w]
    
    # modify scores into integers starting at 0
    scores2 = copy.copy(scores) + 2
    scores2 = np.array(scores2,dtype = 'int8')
    RGB = palette[scores2]
    
    fig, axes = plt.subplots(figsize=figsize)
    im = axes.imshow(RGB, extent = [0,np.shape(scores)[1],-.5,np.shape(scores)[0]], aspect=70,cmap='jet',interpolation = 'none')
    axes.invert_yaxis()
    axes.set_xlabel('frame (5 fps)')
    axes.set_ylabel('worm track #')
    yticks = np.arange(w_tick_start,np.shape(scores2)[0]+w_tick_start,round((np.shape(scores2)[0]+w_tick_start-w_tick_start)/10))
    yticks_pos = yticks - yticks[0]
    axes.set_yticklabels(yticks)
    axes.set_yticks(yticks_pos)
    axes.set_title(title)
    axes.set_aspect(50)
    
    patch0 = mpatches.Patch(color=palette[2]/255, label='quiescent')
    patch1 = mpatches.Patch(color=palette[3]/255, label='cruising')
    patch2 = mpatches.Patch(color=palette[4]/255, label='waving')
    patch3 = mpatches.Patch(color=palette[5]/255, label='standing')
    patch4 = mpatches.Patch(color=palette[1]/255, label='censored')
    
    all_handles = (patch0, patch1, patch2, patch3, patch4)
    
    leg = axes.legend(handles=all_handles,frameon=False,loc='lower right')
    axes.add_artist(leg)
    
    
# USED
def bar_count(dataframe):
    data_num = list(dataframe['manual_behavior_label'])
    data_str = []
    for wf in range(len(data_num)):
        # if data_num[wf] == -2:
        #     data_str.append('not_scored')
        if data_num[wf] == -1:
            data_str.append('censored')
        elif data_num[wf] == 0:
            data_str.append('quiescent')
        elif data_num[wf] == 1:
            data_str.append('crawling')
        elif data_num[wf] == 2:
            data_str.append('waving')
        elif data_num[wf] == 3:
            data_str.append('standing')
    
    
    palette_sns = {'quiescent':palette[2]/255,'crawling':palette[3]/255,
                   'waving':palette[4]/255, 'standing':palette[5]/255, 
                   'censored':palette[1]/255}
    order = ['quiescent','crawling','waving','standing','censored'] 
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data_str,order = order,ax=ax,label="Count",palette = palette_sns)
    plt.show()
    
    
def metric_histogram(dataframe):
    palette = np.array([[  255,   255,   255],   # white - no score
                    [125,   125,   125],   # gray - censored 
                    [  0, 0, 80],   # dark blue - quiescent
                    [  0, 0, 255],   # blue - crawling
                    [  255,   0, 0],   # red - nictating
                    [80, 0, 0]])  # dark red - standing
    
    hist_colors = palette[[3,4,5,2,1]]
    behaviors = dataframe.behavior.unique()
    metrics = dataframe.columns[2:]
    behavior_list = dataframe.behavior
    units = ['pixels','microns','microns','a.u.','radians',
             'microns','microns','radians','microns','microns']
    
    fig, axs = plt.subplots(2,5,figsize=(10,6),tight_layout=True)
    indr = 0
    indc = 0
    ind = 0
    for m in metrics:
        
        metric_list = np.array(dataframe[m])
        cats = []
        for b in behaviors[0:2]:
            cats.append(metric_list[np.where(behavior_list==b)])
        
        e0 = np.min((np.nanmin(cats[0]),np.nanmin(cats[1])))
        ee = np.max((np.nanmax(cats[0]),np.nanmax(cats[1])))
        bin_edges = np.linspace(e0,ee,16) 
        
        for b in range(len(behaviors[0:2])):
            axs[indr,indc].hist(cats[b], bins=bin_edges, alpha=0.5, label=behaviors[b],
                     density = True, color = hist_colors[b]/255)
        #axs[ind].xlabel("Data", size=14)
        #axs[ind].ylabel("Count", size=14)
        axs[indr,indc].title.set_text(m)
        axs[indr,indc].set(xlabel=units[ind])
        if indc in [0,1,2,3]:
            indc = indc+1
        else:
            indc = 0
            indr = indr+1
        ind = ind+1
    plt.show()
    


def scattermatrix(dataframe):
    metrics = dataframe.columns[2:]
    behaviors = dataframe.behavior.unique()
    behavior_list = dataframe.behavior
    sp_colors = palette[[3,4,5,2,1]]
    fig, axs = plt.subplots(len(metrics)-1,len(metrics)-1,figsize=(15,15),tight_layout=True)
    fig.suptitle('Scattermatrix of Nictation Indicators')
    indr = 0
    indc = 0
    ind = 0
    for m1 in np.arange(len(metrics)-1):
        for m2 in np.arange(1,len(metrics)):
            if m2-1-m1 >= 0:
                metric_list_1 = np.array(dataframe[metrics[m1]])
                metric_list_2 = np.array(dataframe[metrics[m2]])
                cats_1 = []
                cats_2 = []
                for b in behaviors:
                    cats_1.append(metric_list_1[np.where(behavior_list==b)])
                    cats_2.append(metric_list_2[np.where(behavior_list==b)])
                
                for b in range(len(behaviors)):
                    axs[m2-1,m1].scatter(cats_1[b],cats_2[b], color = sp_colors[b]/255,
                            alpha=0.3, marker = '.',s=1)
                    if m1 == 0:
                        axs[m2-1,m1].set(ylabel=metrics[m2])
                    if m2-1 == 8:
                        axs[m2-1,m1].set(xlabel=metrics[m1])
            else:
                axs[m2-1,m1].axis('off')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=palette[3]/255, lw=10),
                    Line2D([0], [0], color=palette[4]/255, lw=10),
                    Line2D([0], [0], color=palette[2]/255, lw=10),
                    Line2D([0], [0], color=palette[5]/255, lw=10),
                    Line2D([0], [0], color=palette[1]/255, lw=10)]
    labels = ['cruising','waving','quiescent','standing',
              'censored']
    
    fig.legend(custom_lines, labels,loc = 'right')
    plt.show()
