# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:42:03 2021


This module contains functions for calculating the following nictation metrics
based on their definitions in Lee et al 2012. "nictation ratio was determined
as the nictation time divided by the observation time" Lee et al 2012
In methods: "To exclude dauers in quiescence, nictation was only evaluated
# for consistently moving worms." Lee et al 2012

1. nictation ratio

2. average duration (of nictation bouts)

3. initiation index (rate)


Additionally, because average duration is likely to be scewed due to longer
bouts being but off by tracking failures, a "stopping rate" is also 
calculated.

Issues and improvements:
    
    -


@author: PDMcClanahan
"""

import numpy as np


def nictation_ratio(scores, only_active = True, binary = True):
    '''takes a list of nictation scores and finds the number of frames scored
    as nictation and divides that by the number of frames scored (but not 
    censored).  If <only_active> is true, only waving and crawling are 
    counted. NB: "initiation index was the frequency with which nictation was
    started while dauers cruised the micro-dirt chip"- Lee et al 2012
    '''
    frames_nictating = 0
    frames_scored = 0
    
    if not binary:
        for ws in scores:
            
            for f in range(len(ws)):
                
                if only_active:
                    
                    if ws[f] == 2:
                        frames_nictating += 1
                        frames_scored += 1
                    elif ws[f] == 1:
                        frames_scored += 1
                else:
                    
                    if ws[f] == 2 or ws[f] == 3:
                        frames_nictating += 1
                        frames_scored += 1
                    elif ws[f] == 1 or ws[f] == 0:
                        frames_scored += 1
    else:
        for ws in scores:
            for f in range(len(ws)):
                if ws[f] == 1:
                    frames_nictating += 1
                    frames_scored += 1
                elif ws[f] == 0:
                    frames_scored += 1

    
    
    if frames_scored == 0:
        nict_ratio = np.nan
    else:
        nict_ratio = frames_nictating / frames_scored
    
    return nict_ratio # unitless


def initiation_rate(scores, only_active = True, fps = 5, binary = True):
    '''takes a list of nictation behavior scores and finds the number of 
    transitions from not nictating to nictating divided by the total time not
    nictating. If <only_active> is true, only transitions from crawling to 
    waving are counted, and only crawling is counted as time not nictating.'''
    
    if not binary:
        trans_mat = np.zeros((5,5))
        for wt in range(len(scores)):
            for f in range(len(scores[wt])-1):
                trans_mat[int(scores[wt][f]+1),int(scores[wt][f+1]+1)] += 1
        
        if only_active:
            num = trans_mat[2,3]
            denom = np.sum(trans_mat[2,:])
            
        else:
            num = trans_mat[2,3] + trans_mat[2,4] + trans_mat[1,3] + trans_mat[1,4]
            denom = np.sum(trans_mat[1:3,:])
    else:
        
        trans_mat = np.zeros((3,3))
        for wt in range(len(scores)):
            for f in range(len(scores[wt])-1):
                trans_mat[int(scores[wt][f]+1),int(scores[wt][f+1]+1)] += 1
        num = trans_mat[1,2]
        denom = np.sum(trans_mat[1,:])
        
    if denom != 0:
        IR = fps * (num / denom)
    else:
        IR = np.nan
    
    return IR
    


def stopping_rate(scores, only_active = True, fps = 5, binary = True):
    '''takes a list of nictation behavior scores and finds the number of 
    transitions from nictating to not nictating divided by the total time
    nictating. If <only_active> is true, only transitions from waving to 
    crawling are counted, and only waving is counted as time nictating.''' 
    
    if not binary:
        trans_mat = np.zeros((5,5))
        for wt in range(len(scores)):
            for f in range(len(scores[wt])-1):
                trans_mat[int(scores[wt][f]+1),int(scores[wt][f+1]+1)] += 1
        
        if only_active:
            num = trans_mat[3,2]
            denom = np.sum(trans_mat[3,:])
            
        else:
            num = trans_mat[3,2] + trans_mat[4,2] + trans_mat[3,1] + trans_mat[4,1]
            denom = np.sum(trans_mat[3:5,:])
    else:
        trans_mat = np.zeros((3,3))
        for wt in range(len(scores)):
            for f in range(len(scores[wt])-1):
                trans_mat[int(scores[wt][f]+1),int(scores[wt][f+1]+1)] += 1
        num = trans_mat[2,1]
        denom = np.sum(trans_mat[2,:])
        
    if denom != 0:
        SR = fps * (num / denom)
    else:
        SR = np.nan
    
    return SR


def nictation_duration(scores, exclude_partial_episodes = False, 
                       only_active = True, fps = 5):
    '''Nictation duration is the average duration of nictation bouts.  This
    function returns the average duration of a nictation bout as well as all
    the durations of all the bouts counted.  If <include_partial_episodes> is
    True, then nictation bouts that have begun or are ongoing at the beginning
    or end of a tracking period are counted.  If <only_active> is True, only
    nictation bouts in which there is not standing are counted.  This meansure
    is problematic because, among other reasons, many bouts outlast the period
    of tracking, and this is more likely to happen with longer bouts.'''
    
    
    if exclude_partial_episodes and not only_active:
        episode_durs = []
        
        for ws in scores:
            search = True
            
            for f in range(1,len(ws)):
                if search:
                    if ws[f-1] in (0,1) and ws[f] in (2,3):
                        search = False
                        dur = 1
                else:
                    if ws[f-1] in (2,3) and ws[f] in (2,3):
                        dur += 1
                    else:
                        episode_durs.append(dur)
                        search = True
        
        if len(episode_durs) > 0:
            episode_durs = np.array(episode_durs) * (1/fps)
            nict_dur = np.mean(episode_durs)
        else:
            nict_dur == np.nan
            episode_durs = np.nan
            
        return nict_dur, episode_durs
            
    
    elif exclude_partial_episodes and only_active:
        episode_durs = []
        
        for ws in scores:
            search = True
            
            for f in range(1,len(ws)):
                if search:
                    if ws[f-1] == 1 and ws[f] == 2: # crawling to waving
                        search = False
                        dur = 1
                else:
                    if ws[f-1] == 2 and ws[f] == 2: # waving continues
                        dur += 1
                    elif ws[f-1] == 2 and ws[f] == 1: # waving to crawling
                        episode_durs.append(dur)
                        search = True
                    else: # any other scenario - ongoing episode is discarded
                        search = True
                        
        if len(episode_durs) > 0:
            episode_durs = np.array(episode_durs) * (1/fps)
            nict_dur = np.mean(episode_durs)
        else:
            nict_dur == np.nan
            episode_durs = np.nan
        
        return nict_dur, episode_durs
            
        
    elif not exclude_partial_episodes and not only_active:
        
        tot_f_nict = 0
        tot_nict_init = 0
        
        for ws in scores:
            tot_f_nict += np.sum(ws==2) + np.sum(ws==3)
        
            for f in range(1,len(ws)):
                if ws[f-1] in (0,1) and ws[f] in (2,3):
                    tot_nict_init += 1
                    
        if tot_nict_init > 0:
            nict_dur =  (tot_f_nict*(1/fps)) / tot_nict_init
        else:
            nict_dur = np.nan
        
        return nict_dur, np.nan
            
    elif not exclude_partial_episodes and only_active:
        
        tot_f_nict = 0
        tot_nict_init = 0
        
        for ws in scores:
            tot_f_nict += np.sum(ws==2)
        
            for f in range(1,len(ws)):
                if ws[f-1] == 1 and ws[f] == 2:
                    tot_nict_init += 1
                    
        if tot_nict_init > 0:
            nict_dur = (tot_f_nict*(1/fps)) / tot_nict_init
        else:
            nict_dur = np.nan
        
        return nict_dur, np.nan
            
        
    
    


