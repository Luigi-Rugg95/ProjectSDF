# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:04:12 2022

@author: Luigi
"""

import numpy as np
import time

#importing functions
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask

segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])


#segmentation = np.array([[1,0,0],[1,1,0],[1,1,1]])

start = time.time()
f = sdf_mask(segmentation,0.01)
plotting(f.sdf(),*f.grid())
end=time.time()
print(end-start)

grid_points = np.array([0,0])
poly = np.array([1,1])
print(f.diff_point_array(grid_points,poly))
