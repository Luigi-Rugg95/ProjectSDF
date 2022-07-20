# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 09:54:02 2022

@author: Luigi
"""

import numpy as np
from plotting import plotting
from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from sdf_from_binary_mask import *

"""
----------
Function utilities for testing
----------
"""



def utility_iterate_shapes(segmentation): 
    """
    

    Parameters
    ----------
    segmentation : numpy.ndarray
        Input binary mask

    Returns
    -------
    separeted_pol : list bolean type
        length of the list corresponds to the number of shapes found in the segmentation

    """
    separeted_pol = [shape for shape in iterate_shapes(segmentation)]
    return separeted_pol
    
def utility_generate_sides(segmentation): 
    """
    

    Parameters
    ----------
    segmentation : numpy.ndarray
        Input binary mask

    Returns
    -------
    sides : list of float
        all the sides of the shape found in segmentation

    """
    
    sides = [((p1,p2)) for p1,p2 in generate_sides(shape_as_points(segmentation))]
    return sides

def utility_distance_from_poly_1(grid_finess):
    """
    

    Parameters
    ----------
    grid_finess : float

    Returns
    -------
    integer
        number of points inside the shape*
    integer
        number of points along one side of the shape*

    Description
    -----------
    This function calculates (analytical calculation) only for a unitary square cube (*) 
    the number of points of a grid given a specific grid_finess which lays inside the square, 
    and the number of points which lay along the side of the square
    """
    X, Y = np.mgrid[-1:1+grid_finess:grid_finess,-1:1+grid_finess:grid_finess]
    x = Y[0] 
    points_inside = x[abs(x)<0.5]
    return (points_inside.size)**2

def utility_distance_from_poly_2(grid_finess):
    """
    

    Parameters
    ----------
    grid_finess : float

    Returns
    -------
    integer
        number of points inside the shape*
    integer
        number of points along the side of the shape*

    Description
    -----------
    This function calculates (analytical calculation) only for a shape made of two unitary square cubes (*) 
    the number of points of a grid given a specific grid_finess which lays inside the square, 
    and the number of points which lay along the side of the shape
    """
    
    x = np.linspace(-1,2,int((3/grid_finess+1)))
    y = np.linspace(-1,1,int((2/grid_finess+1)))
    
    points_inside_x = x[(x>-0.5) & (x<1.5)]
    points_inside_y = y[abs(y)<0.5]
    
    points_along_x = x[(x==-0.5) | (x ==1.5)]
    points_along_y = y[abs(y)==0.5]
    
    return (points_inside_x.size*points_inside_y.size,points_along_x.size/2*(points_inside_x.size+2)*2+points_along_y.size/2*(points_inside_y.size+2)*2-4)
    
