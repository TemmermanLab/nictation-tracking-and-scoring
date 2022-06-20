# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:42:03 2021


This file contains functions for calculating the following nictation metrics
based on their definitions in Lee et al 2012:

1. nictation ratio

2. average duration (of nictation bouts)

3. initiation index (rate)


@author: PDMcClanahan
"""
import numpy as np
import pdb

testing = False

# "nictation ratio was determined as the nictation time divided by the
# observation time" Lee et al 2012
# in methods: "To exclude dauers in quiescence, nictation was only evaluated
# for consistently moving worms." Lee et al 2012
# "only_active TRUE" uses active (cruising) time as the denominator,
# "only_active FALSE" uses quiescent + active (recumbent) time as the
# denominator. Both use the sum of waving and standing nictation as the
# numerator.
def nictation_ratio(scores, only_active = True):
    #pdb.set_trace()
    if type(scores)==list:
        scores_comb = np.array([])
        for ws in scores: scores_comb = np.concatenate((scores_comb,ws))
    else:
        scores_comb = scores
    
    if only_active:
        tot_f_nict = np.sum(scores_comb==2) + np.sum(scores_comb==3)
        tot_f_recumb = np.sum(scores_comb==1)
    
    else:
        tot_f_nict = np.sum(scores_comb==2) + np.sum(scores_comb==3)
        tot_f_recumb = np.sum(scores_comb==1) + np.sum(scores_comb==0)
    
    return tot_f_nict/(tot_f_recumb+tot_f_nict)


# aka initiation index; "initiation index was the frequency with which
# nictation was started while dauers cruised the micro-dirt chip"- Lee et al
# 2012. "only_active TRUE" uses active (cruising) time as the denominator, and
# only considers cruising to (active or quiescenct) nictation transitions.
# "only_active FALSE" uses quiescent + active (recumbent) time as the
# denominator and considers all recumbent to nictation transitions.
# Returns initiation rate in Hertz
def initiation_rate(scores, only_active = True, fps = 5):
    
    if only_active:
        scores_comb = np.array([])
        for ws in scores: scores_comb = np.concatenate((scores_comb,ws))
        tot_f_cruise = np.sum(scores_comb==1)
        
        tot_nict_init = 0
        for w in range(len(scores)):
            for f in range(1,len(scores[w])):
                if scores[w][f-1] == 1 and scores[w][f] in (2,3):
                    tot_nict_init += 1
        
        init_rate = tot_nict_init / tot_f_cruise
        
    else:
        scores_comb = np.array([])
        for ws in scores: scores_comb = np.concatenate((scores_comb,ws))
        tot_f_recumb = np.sum(scores_comb==0) + np.sum(scores_comb==1)
        
        tot_nict_init = 0
        for w in range(len(scores)):
            for f in range(1,len(scores[w])):
                if scores[w][f-1] in (0,1) and scores[w][f] in (2,3):
                    tot_nict_init += 1
        
        init_rate = tot_nict_init / tot_f_recumb
        
    return init_rate * (1/fps)


def nictation_duration(scores, exclude_partial_episodes = False, fps = 5):
    #import pdb; pdb.set_trace()
    if exclude_partial_episodes:
        episode_durs = []
        
        for w in range(len(scores)):
            search = True
            for f in range(1,len(scores[w])):
                if search:
                    if scores[w][f-1] in (0,1) and scores[w][f] in (2,3):
                        search = False
                        dur = 1
                else:
                    if scores[w][f-1] in (2,3) and scores[w][f] in (2,3):
                        dur += 1
                    else:
                        episode_durs.append(dur)
                        search = True
            
        #import pdb; pdb.set_trace()
        episode_durs = np.array(episode_durs) * (1/fps)
        nict_dur = np.mean(episode_durs)
        
        
    else:
        scores_comb = np.array([])
        for ws in scores: scores_comb = np.concatenate((scores_comb,ws))
        tot_f_nict = np.sum(scores_comb==2) + np.sum(scores_comb==3)
        
        tot_nict_init = 0
        for w in range(len(scores)):
            for f in range(1,len(scores[w])):
                if scores[w][f-1] in (0,1) and scores[w][f] in (2,3):
                    tot_nict_init += 1
        
        episode_durs = False
        nict_dur = tot_f_nict / tot_nict_init
    
    return nict_dur, episode_durs




###############################################################################

# testing
if testing: 
    # calculate nictation metrics
    behavior_scores = pickle.load(open(r'E:\20210212_Cu_ring_test\dauers 14-14-56_tracking\051821_quaternary_scoring\manual_nictation_scores.p','rb')) 
    
    
    nrat1 = nictation_ratio(behavior_scores,True)
    nrat2 = nictation_ratio(behavior_scores,False)
    
    initr1 = initiation_rate(behavior_scores, True)
    initr2 = initiation_rate(behavior_scores, False)
    initr3 = initiation_rate(behavior_scores, False, 10)
    
    avg_dur1,dur1 = nictation_duration(behavior_scores)
    avg_dur2,dur2 = nictation_duration(behavior_scores,True)