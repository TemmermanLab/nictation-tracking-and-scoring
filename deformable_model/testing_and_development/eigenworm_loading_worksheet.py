# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:53:34 2022

@author: PDMcClanahan
"""

import os
import pickle
import matplotlib.pyplot as plt


eigenworm_file = os.path.split(os.path.split(__file__)[0])[0] + '\\testing_files\\20211212_Sc_eigenworms.p'
n_coeff = 5

with open(eigenworm_file, 'rb') as f:
    eigendict = pickle.load(f)
    M = eigendict['matrix']
    EVals = eigendict['eigenvalues'][0:n_coeff]
    EVecs = eigendict['eigenvectors'][-n_coeff:-1]


evec = EVecs[0]
plt.plot(evec)
