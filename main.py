# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:04:12 2022

@author: Luigi
"""

import random as rm
import numpy as np

#importing functions
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask

segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
])


#segmentation = np.array([[1,0],[0,1]])

segmentation = np.array([[1],])


f = sdf_mask(segmentation, 1)
#print(f.grid()[0])
plotting(f.sdf(),*f.grid())

#print(f.utility_generate_sides()[0])