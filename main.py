# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:04:12 2022

@author: Luigi
"""

import numpy as np

#importing functions
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from sdf_from_binary_mask import load_segmentation
"""
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
"""

segmentations, files_name = load_segmentation()

for segmentation in segmentations:
    twodsdf = sdf_mask(segmentation,0.1) #parameters: segmentation, grid_finess
    plotting(twodsdf.sdf(),*twodsdf.grid(), "y") #parameters: sdf, *grid, if "y" it saves the figure
