# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:04:12 2022

@author: Luigi
"""

import numpy as np

#importing functions
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from sdf_from_binary_mask import *

#load the segmentation and the name of the input
segmentations, files_name = load_segmentation()


for segmentation,file_name in zip(segmentations,files_name):
    twodsdf = sdf_mask(segmentation,0.1) #parameters: segmentation, grid_finess
    twodsdf.write_sdf(twodsdf.sdf(), file_name) #file_name is also given in order to keep track of it in the output
    plotting(twodsdf.sdf(),*twodsdf.grid()) #parameters: sdf, *grid

