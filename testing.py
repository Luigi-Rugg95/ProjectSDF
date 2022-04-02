# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:43:42 2022

@author: Luigi
"""

#from functions import final_distances,distance_from_poly,diff_point_array,merge_cubes,generate_sides,shape_as_points,iterate_shapes

from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask

import numpy as np
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from datetime import timedelta
import pytest


#@given(grid_finess=st.floats())
def test_sdf_init(): 
    """
    

    Parameters
    ----------
    segmentation : numpy.ndarray
        binary mask given as initial input for calculating the sdf
    grid_finess : float
        finess of the grid

    Testing
    -------
    correct dimension of the binary mask (2D)
    finesse of the grid
    correct limits of the grid
    correct dimensions of the arrays
    limiting case of no empty binary mask

    """
    segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],])

    grid_finess = 0.1
    
    test_sdf = sdf_mask(segmentation,grid_finess)
    
    #2D sdf checking
    if np.size(test_sdf.segmentation.shape)!=2: pytest.raises(AssertionError)   
    
    #at least grid finess smaller than the smallest pixel    
    if test_sdf.grid_finess>1: pytest.raises(AssertionError)
    if len(segmentation[segmentation!=0])==0: pytest.raises(AssertionError)    
    
    #extension of the grid, in order to cover the whole segmentation
    assert(np.max(test_sdf.grid()[0])>segmentation[:,0].size)
    assert(np.max(test_sdf.grid()[1])>segmentation[0].size)
    assert(np.min(test_sdf.grid()[0])<0)
    assert(np.min(test_sdf.grid()[1])<0)
    

def test_iterate_shapes(): 
    """
    

    Testing
    -------
    number of shapes obtained corresponds to the real one

    """
    
    #to be ended, try to parametrize the segmentation in order to get one with known shape
    
    segmentation= np.array([
    [0, 1, 0, 1, 0, 1, 0, 1]
    ,])            
    
    grid_finess=0.1
    
    test_sdf = sdf_mask(segmentation,grid_finess)
    separeted_pol = [shape for shape in test_sdf.iterate_shapes(segmentation)]
    assert len(separeted_pol) == 4
    

def test_shape_as_points(): 
    """
    

    Testing
    -------
    centre of the pixel returned is within segmentation boundaries
    
    """
    
    
    segmentation = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],])

    grid_finess=0.1
    
    
    test_sdf = sdf_mask(segmentation,grid_finess)
    
    assert np.max(test_sdf.shape_as_points(segmentation)[:,0])<=segmentation[:,0].size
    assert np.max(test_sdf.shape_as_points(segmentation)[:,1])<=segmentation[0].size
    assert np.min(test_sdf.shape_as_points(segmentation)[:,0])>=0
    assert np.min(test_sdf.shape_as_points(segmentation)[:,1])>=0
    
    
def test_generate_sides(): 
    
    return


