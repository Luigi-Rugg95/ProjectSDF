# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 09:54:02 2022

@author: Luigi
"""

import numpy as np
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask


"""
----------
Function utilities for testing
----------
"""



def utility_iterate_shapes(segmentation): 
    test_sdf=sdf_mask(segmentation,1)
    separeted_pol = [shape for shape in test_sdf.iterate_shapes(segmentation)]
    return separeted_pol
    
def utility_generate_sides(segmentation): 
    test_sdf=sdf_mask(segmentation,1)
    sides = [((p1,p2)) for p1,p2 in test_sdf.generate_sides(test_sdf.shape_as_points(segmentation))]
    return sides

def utility_distance_from_poly_1(grid_finess):
    x = np.linspace(-1,1,int((2/grid_finess+1)))
    points_inside = x[abs(x)<0.5]
    points_along = x[abs(x)==0.5]
    return (points_inside.size)**2,points_along.size/2*(points_inside.size+1)*4

def utility_distance_from_poly_2(grid_finess):
    x = np.linspace(-1,2,int((3/grid_finess+1)))
    y = np.linspace(-1,1,int((2/grid_finess+1)))
    
    points_inside_x = x[(x>-0.5) & (x<1.5)]
    points_inside_y = y[abs(y)<0.5]
    
    points_along_x = x[(x==-0.5) | (x ==1.5)]
    points_along_y = y[abs(y)==0.5]
    
    return (points_inside_x.size*points_inside_y.size,points_along_x.size/2*(points_inside_x.size+2)*2+points_along_y.size/2*(points_inside_y.size+2)*2-4)
    
    