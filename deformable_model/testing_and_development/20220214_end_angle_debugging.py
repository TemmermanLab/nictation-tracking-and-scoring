# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:06:13 2022

@author: Temmerman Lab
"""

plt.plot(xs[280:290],ys[280:290],'r.')

a_raw = np.array([-np.pi/2,-np.pi/2,-(3*np.pi)/4,np.pi,-(3*np.pi)/4,(3*np.pi)/4,-(3*np.pi)/4,np.pi,np.pi])
da = np.diff(a_raw,n=1,axis=0)
#da_u = np.unwrap(da)
#da_u2 = np.unwrap(da,discont = (7.5*np.pi)/4)

da_unwr = copy.copy(da)
for a in range(len(da_unwr)):
    if da_unwr[a] > np.pi:
        da_unwr[a]  = -(2*np.pi-da_unwr[a] )
    elif da_unwr[a] < -np.pi:
        da_unwr[a] = 2*np.pi+da_unwr[a]
        
        

