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

#Defining a Shape
"""
poly = [
    [0, 0],
    [2, 0],
    [2, 1],
    [1, 1.5],
    [0, 1],
    [1, 0.5],
]
"""
#Defining a random shape

"""
poly = [[1,1]]

n=10

for i in range(0,n): 
    poly.append([np.sqrt(rm.randint(1,5))*np.cos(i*2*np.pi/(n-1)),np.sqrt(rm.randint(1,5))*np.sin(i*2*np.pi/(n-1))])    
poly=poly[1:]    
"""

#Grid from which we calculate the distance

"""
----------------
Binary Mask Blob
----------------

"""
"""
segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
"""
segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
])

grid_finess = 0.1

limit_grid = [segmentation[:,0].size, segmentation[0].size]
#creating a meshgrid
X, Y = np.mgrid[-1:limit_grid[0]+1:grid_finess,-1:limit_grid[1]+1:grid_finess]
XY = np.dstack([X, Y])
points_to_sample = XY.reshape(-1, 2)

#segmentation= np.array([[0],])            

    
f = sdf_mask(segmentation, 0.1)


plotting(f.sdf(),*f.grid())
