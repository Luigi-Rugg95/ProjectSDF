# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:04:12 2022

@author: Luigi
"""

import numpy as np
from functions_for_calculating_distances import distance_from_poly, sdf_complex_gradient
from plotting import plotting_shape


#Defining a Shape
poly = [
    [0, 0],
    [2, 0],
    [2, 1],
    [1, 1.5],
    [0, 1],
    [1, 0.5],
]

#Points from which we calculate the distance
X, Y = np.mgrid[-1:3:0.01, -1:3:0.01]
XY = np.dstack([X, Y])
p = XY.reshape(-1, 2)
#print(XY.shape)



distances = distance_from_poly(poly, p)
distances = distances.reshape(*XY.shape[:-1])
edges_xn, edges_yn, gradient_norm = sdf_complex_gradient(distances)

plotting_shape(distances,edges_xn,edges_yn,gradient_norm, X, Y, poly)