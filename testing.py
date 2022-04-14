# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:43:42 2022

@author: Luigi
"""

#from functions import final_distances,distance_from_poly,diff_point_array,merge_cubes,generate_sides,shape_as_points,iterate_shapes

from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask

import numpy as np
from scipy.ndimage import label, generate_binary_structure

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from datetime import timedelta
import pytest


def test_sdf_init_grid_finess(): 
    """

    Testing
    -------
    Unit test on finesse of the grid limit values (0,1)
    for having a good resolution    
    
    Using a square of unitary length as input
    
    """
    segmentation = np.array([[1],])
    
    grid_finess=2
    with pytest.raises(AssertionError):
        assert sdf_mask(segmentation,grid_finess)
        
    
    

def test_grid():
    """
    

    Testing
    -------
    testing the grid creating
    limits of the grid contains the entire figure
    grid finess finite value
    
    Using a square of unitary length as input
    
    """
    segmentation = np.array([[1],])
    grid_finess=0.1
    
    test_sdf = sdf_mask(segmentation,grid_finess)
    assert(np.max(test_sdf.grid()[0])>=segmentation[:,0].size)
    assert(np.max(test_sdf.grid()[1])>=segmentation[0].size)
    assert(np.min(test_sdf.grid()[0])<0)
    assert(np.min(test_sdf.grid()[1])<0)
        
    
    
    grid_finess=0   
    test_sdf = sdf_mask(segmentation,grid_finess)
    with pytest.raises(ZeroDivisionError):
           assert test_sdf.grid()

    
def test_init_segmentation(): 
    
    #these are all unit test
    
    segmentation=np.array([[[0,1,0],],])
    grid_finess= 0.1
    with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)
    
    segmentation=np.array([0,1,0])
    grid_finess= 0.1
    with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)
    
    segmentation=np.array([[0],])
    with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)



"""
Unit testing using a unitary square centered in (0,0) as input labelled with 1
"""

@pytest.fixture
def unitary_cube(): 
    return [np.array([[1],]),0.1]
    


def test_iterate_shapes_1(unitary_cube): 
    """
    Parameters
    ----------
    segmentation = unitary cube[0] 
    grid_finess = unitary cube[1]

    Testing
    -------
    Output of the function generator
    
    """
    
    test_sdf = sdf_mask(*unitary_cube)
    #using function utility used
    
    assert len(test_sdf.utility_iterate_shapes()) == 1     
    assert test_sdf.utility_iterate_shapes()[0] == [[True]]     

def test_shape_as_points_1(unitary_cube): 
    """
    Parameters
    ----------
    segmentation = unitary cube[0] 
    grid_finess = unitary cube[1]

    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*unitary_cube)
    assert test_sdf.shape_as_points(unitary_cube[0]).shape[0] == 1
    assert test_sdf.shape_as_points(unitary_cube[0]).shape[1] == 2
    assert test_sdf.shape_as_points(unitary_cube[0]).all() == 0


def test_generate_sides_1(unitary_cube): 
    """
    Parameters
    ----------
    segmentation = unitary cube[0] 
    grid_finess = unitary cube[1]

    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*unitary_cube)
    assert len(test_sdf.utility_generate_sides()) == 4
    
     

@given(segmentation = st.lists(st.tuples(st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1))))
def test_iterate_shapes(segmentation): 
    """
    Parameters 
    ----------    
    segmentetion: numpy.array
        1D binary mask
    
    Testing
    -------
    number of shapes obtained corresponds to the real one
    """
    
    #to be ended, try to parametrize the segmentation in order to get one with known shape
    
    segmentation= np.array(segmentation)            
    grid_finess = 0.1
        
    if len(segmentation[segmentation!=0])==0:
        with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)
    else:
        num_shapes = label(segmentation)[1]
        test_sdf = sdf_mask(segmentation,grid_finess)
        separeted_pol = [shape for shape in test_sdf.iterate_shapes(segmentation)]
        assert len(separeted_pol) == num_shapes
        
@given(segmentation = st.lists(st.tuples(st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,1))))

def test_shape_as_points(segmentation): 
    """
    

    Testing
    -------
    centres of the pixel returned is within segmentation boundaries
    coordinates returned in 2D
    """
    
    segmentation = np.array(segmentation)
    
    grid_finess=0.1
    
    #segmentation = np.array([[0],])
    
    if len(segmentation[segmentation!=0])==0:
        with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)
    
    
    else: 
        test_sdf = sdf_mask(segmentation,grid_finess)    
        assert np.max(test_sdf.shape_as_points(segmentation)[:,0])<=segmentation[:,0].size
        assert np.max(test_sdf.shape_as_points(segmentation)[:,1])<=segmentation[0].size
        assert np.min(test_sdf.shape_as_points(segmentation)[:,0])>=0
        assert np.min(test_sdf.shape_as_points(segmentation)[:,1])>=0
        assert test_sdf.shape_as_points(segmentation).shape[1]==2
        assert(np.all(test_sdf.shape_as_points(segmentation) >= 0))


@given(points = st.lists(st.tuples(st.integers(0,10),st.integers(0,10)),unique=True))
    
def test_generate_sides(points): 
    
    segmentation = np.array([[0,1],])
    grid_finess = 0.1
    test_sdf = sdf_mask(segmentation,grid_finess)
    
    
    points = np.array(points)
    
    
    if np.size(points.shape) < 2:  #the previous tests assure already the right shape 
        return
    
    elif points.shape[1]!=2: 
        with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)
    
    else:
        sides_duplicated = {s for s in test_sdf.generate_sides(points)}
        assert len(sides_duplicated)==4*points.shape[0] 
    
    
    return


def test_merge_cubes(): 
    
    segmentation= np.array([
    [0, 1, 0, 1, 0, 1, 0, 1]
    ,])            
    
    grid_finess = 0.1
    
    test_sdf = sdf_mask(segmentation,grid_finess)

