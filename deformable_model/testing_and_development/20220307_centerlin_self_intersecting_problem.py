# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:41:18 2022

@author: Temmerman Lab
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def counter_clockwise(p1,p2,p3):
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

# Returns true if line segments p1p2 and p3p4 intersect
def intersect(p1,p2,p3,p4):
    return counter_clockwise(p1,p3,p4) != counter_clockwise(p2,p3,p4) \
        and counter_clockwise(p1,p2,p3) != counter_clockwise(p1,p2,p4)

def self_intersect(cline):
    self_int = False
    for s1 in range(len(cline[0])-1):
        p1 = cline[:,s1]; p2 = cline[:,s1+1]
        s2s = np.arange(len(cline[0])-1).astype(np.uint8)
        s2s = np.delete(s2s,np.where(np.logical_and(s2s >= s1-1,
                                                    s2s <= s1+1))[0])
        for s2 in s2s:
            p3 = cline[:,s2]; p4 = cline[:,s2+1]
            if intersect(p1,p2,p3,p4):
                self_int = True; break
        if self_int:
            break
    
    return self_int


# self-intersecting example
xs = [1523.0, 1517.7, 1512.4, 1507.2, 1502, 1496.7, 1491.2, 1486.1, 1485.5, 
      1489.7, 1494.8, 1500.4, 1506.6, 1513.1, 1518.6, 1523.4, 1527.6, 1529.7,
      1529.5, 1527.1, 1523.0, 1517.8, 1512.0, 1505.5, 1498.7, 1492.3, 1486.5,	
      1480.6, 1474.2, 1467.4, 1460.9, 1455.0, 1450.0, 1446.0, 1442.9, 1440.8, 
      1439.5, 1439.3, 1441.0, 1443.4, 1445.6, 1449.2, 1454.1, 1459.6, 1465.4, 
      1471.4, 1477.2, 1482.8, 1488.4, 1494]
ys = [677.0, 680.9, 684.9, 688.8, 692.7, 696.5, 700.4, 704.7, 710.4, 715.7,
      720.1, 723.4, 725.5, 725.6, 722.5, 717.7, 712.4, 706.5, 700.1, 693.9,
      688.5, 684.4, 682.2, 682.2, 683.0, 684.3, 687.3, 690.1, 690.9, 690.8,
      691.2, 693.6, 697.6, 702.9, 708.8, 715.1, 721.5, 727.8, 734.1, 740.4,
      746.5, 751.9, 756.4, 759.9, 762.9, 765.8, 768.8, 772.1, 775.5, 779.1]


main_pts = np.array([xs,ys])
print(self_intersect(main_pts))
plt.plot(main_pts[0],main_pts[1],'k.')

tck, u = interpolate.splprep([main_pts[0], main_pts[1]],s=0)
u_new = np.linspace(0, 1, 50)
mov_cl = np.array(interpolate.splev(u_new, tck))
self_intersect(mov_cl)
plt.plot(mov_cl[0],mov_cl[1],'k-')



# version extracted from the tracking code
Centerline = np.array([[ 62. , 105.1],
       [ 56.4, 101.5],
       [ 50.8,  98.1],
       [ 45.2,  94.8],
       [ 39.4,  91.8],
       [ 33.4,  88.9],
       [ 27.6,  85.9],
       [ 22.1,  82.4],
       [ 17.2,  77.9],
       [ 13.6,  72.5],
       [ 11.4,  66.4],
       [  9. ,  60.1],
       [  7.3,  53.8],
       [  7.5,  47.5],
       [  8.8,  41.1],
       [ 10.9,  34.8],
       [ 14. ,  28.9],
       [ 18. ,  23.6],
       [ 23. ,  19.6],
       [ 28.9,  17.2],
       [ 35.4,  16.8],
       [ 42.2,  16.9],
       [ 48.6,  16.1],
       [ 54.5,  13.3],
       [ 60.3,  10.3],
       [ 66.7,   9. ],
       [ 73.5,   8.2],
       [ 80. ,   8.2],
       [ 85.8,  10.4],
       [ 91. ,  14.5],
       [ 95.1,  19.9],
       [ 97.5,  26.1],
       [ 97.7,  32.5],
       [ 95.6,  38.4],
       [ 91.4,  43.7],
       [ 86.6,  48.5],
       [ 81.1,  51.6],
       [ 74.6,  51.5],
       [ 68.4,  49.4],
       [ 62.8,  46.1],
       [ 57.7,  41.7],
       [ 53.5,  36.4],
       [ 54.1,  30.7],
       [ 59.2,  26.4],
       [ 64.7,  22.5],
       [ 70. ,  18.7],
       [ 75.2,  14.8],
       [ 80.4,  10.9],
       [ 85.7,   6.9],
       [ 91. ,   3. ]])
Centerline = np.swapaxes(Centerline,0,1)
print(self_intersect(Centerline))


