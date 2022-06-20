# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 06:16:44 2022

@author: PDMcClanahan
"""

from scipy.interpolate import interp1d
f = interp1d(cline_ext[0], cline_ext[1], kind='cubic',bounds_error=False, fill_value='extrapolate')

# cubic spline
xn = np.linspace(cline_ext[0][0],cline_ext[0][-1],250)
plt.plot(cline_ext[0], cline_ext[1],'k.',xn,f(xn),'r-')
plt.show()


# parameteric spline
from scipy import interpolate
tck, u = interpolate.splprep([cline_ext[0], cline_ext[1]], s=0)
unew = np.arange(0, 1.10, 0.01)
out = interpolate.splev(unew, tck)
plt.plot(cline_ext[0], cline_ext[1],'k.',out[0],out[1],'r-')
plt.show()

unew = np.linspace(-.1, 1, 100)
out = interpolate.splev(unew, tck)
plt.plot(cline_ext[0], cline_ext[1],'k-',out[0],out[1],'r.')
plt.show()
