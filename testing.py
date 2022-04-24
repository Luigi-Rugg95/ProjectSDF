# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:43:42 2022

@author: Luigi
"""


from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from function_utilities import utility_distance_from_poly_1,utility_distance_from_poly_2,utility_generate_sides,utility_iterate_shapes

import numpy as np
import pytest

#from scipy.ndimage import label, generate_binary_structure
#from hypothesis import strategies as st
#from hypothesis import given
#from hypothesis import settings
#from datetime import timedelta


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
    with pytest.raises(Exception):
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
    """
    Returns
    -------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    """
    return [np.array([[1],]),0.1]
    


def test_iterate_shapes_1(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
     
    
    Testing
    -------
    Output of the function generator
    
    """
    
    test_sdf = sdf_mask(*unitary_cube)
    #using function utility used
    
    assert len(utility_iterate_shapes(unitary_cube[0])) == 1     
    assert (utility_iterate_shapes(unitary_cube[0])[0] == True).all()     

def test_shape_as_points_1(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*unitary_cube)
    assert test_sdf.shape_as_points(unitary_cube[0]).shape[0] == 1
    assert test_sdf.shape_as_points(unitary_cube[0]).shape[1] == 2
    #assert all(test_sdf.shape_as_points(unitary_cube[0])) == 0


def test_generate_sides_1(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*unitary_cube)
    assert len(utility_generate_sides(unitary_cube[0])) == 4
    assert utility_generate_sides(unitary_cube[0])[0] == ((0.5,0.5),(0.5,-0.5)) 
    assert utility_generate_sides(unitary_cube[0])[1] == ((0.5,-0.5),(-0.5,-0.5)) 
    assert utility_generate_sides(unitary_cube[0])[2] == ((-0.5,-0.5),(-0.5,0.5)) 
    assert utility_generate_sides(unitary_cube[0])[3] == ((-0.5,0.5),(0.5,0.5)) 
     

def test_merge_cubes_1(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    

    Testing
    -------
    For the unitary cube as input, merge_cubes() returns only 
    the cordinate of the corner of the cube
    """
    test_sdf = sdf_mask(*unitary_cube)
    assert len(test_sdf.merge_cubes(unitary_cube[0])) == 4
    assert test_sdf.merge_cubes(unitary_cube[0])[0] == (0.5,0.5)
    assert test_sdf.merge_cubes(unitary_cube[0])[1] == (0.5,-0.5)
    assert test_sdf.merge_cubes(unitary_cube[0])[2] == (-0.5,-0.5)
    assert test_sdf.merge_cubes(unitary_cube[0])[3] == (-0.5,0.5)
    

def test_diff_point_array_1(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Array of distances as vector returned after calculating 
    the difference between one point at the origin and 
    whatever point in the grid, this should always return the coordinate
    of the point of the grid itself
    """    
    
    test_sdf = sdf_mask(*unitary_cube)
    grid_points = np.array([-1,-1])
    origin = np.array([0,0])
    assert(test_sdf.diff_point_array(grid_points,origin) == grid_points).all()
    
    grid_points = np.array([-1,1])
    assert(test_sdf.diff_point_array(grid_points,origin) == grid_points).all()
    
    grid_points = np.array([1,1])
    assert(test_sdf.diff_point_array(grid_points,origin) == grid_points).all()

def test_diff_point_array_2(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Array of distances as vector returned after calculating 
    the difference between one point at the (1,1) and 
    whatever point in the grid
    """    
    
    test_sdf = sdf_mask(*unitary_cube)
    grid_points = np.array([-1,-1])
    point = np.array([1,1])
    assert(test_sdf.diff_point_array(grid_points,point) == [-2,-2]).all()
    
    grid_points = np.array([3,3])
    assert(test_sdf.diff_point_array(grid_points,point) == [2,2]).all()
    
    grid_points = np.array([0,0])
    assert(test_sdf.diff_point_array(grid_points,point) == [-1,-1]).all()

    
    
def test_distance_from_poly_1(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Number of points inside the cube (calculated distance smaller than 0) or 
    along the side (calculated distance equal to 0) for different grid_finess
    
    """
    #grid_finess = 1, only one point inside the cube, no points along the side
    test_sdf = sdf_mask(unitary_cube[0],1)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 1
    assert test_sdf.distances[test_sdf.distances==0].size == 0
    
    
    #grid_finess = 0.5 only one point inside the cube, 8 over along the side
    test_sdf = sdf_mask(unitary_cube[0],0.5)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 1
    assert test_sdf.distances[test_sdf.distances==0].size == 8
    
    #grid_finess = 0.1 81 points inside the cube, and 40 along the side
    test_sdf = sdf_mask(unitary_cube[0],0.1)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 81
    assert test_sdf.distances[test_sdf.distances==0].size == 40
    
    #we can compare this with a theoretical value given by the grid_finess using an utility function
    # which calculates the theoretical number of points inside and along the side of the unitary cube
    test_sdf = sdf_mask(unitary_cube[0],0.02)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == utility_distance_from_poly_1(0.02)[0]
    assert test_sdf.distances[test_sdf.distances==0].size == utility_distance_from_poly_1(0.02)[1]

def test_calculate_distance_1(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Shape of the attribute distances, since we have only one shape,
    we obtain only one set of distance
    the shape should be coherent with the dimension of the grid

    """
    test_sdf = sdf_mask(*unitary_cube)
    test_sdf.sdf()
    assert test_sdf.distances.shape[0]==1
    assert test_sdf.distances.shape[1]==test_sdf.grid()[0].shape[0]
    assert test_sdf.distances.shape[2]==test_sdf.grid()[0].shape[1]
    

def test_sdf_1_1(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Value of the distance obtained
    With a grid_finess = 1 we get only 9 points in the grid.
    The corners points will have the same minimum distance from the poly,
    this is just the one fourth of the diagonal of the square defined by the grid
    The other grid points will have same distance, even though the one at the centre
    will be defined with a negative sign, sign inside the square
    """
    
    test_sdf = sdf_mask(unitary_cube[0],1)
    assert len(test_sdf.sdf()[test_sdf.sdf() ==0.5]) == 4
    assert len(test_sdf.sdf()[test_sdf.sdf() ==-0.5]) == 1
    assert len(test_sdf.sdf()[abs(test_sdf.sdf()) ==np.sqrt(2)/2]) == 4
    
"""
Unit testing using two unitary squares side by side, test labelled with 2
"""

def test_sdf_1_2(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    In the case of one figure the sdf returned is the same as the
    self.distance returned by distance_from_poly
    """
    
    test_sdf = sdf_mask(unitary_cube[0],1)
    assert (test_sdf.sdf()-test_sdf.distances == 0).all()



@pytest.fixture
def two_unitary_cube(): 
    """

    Returns
    -------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    """
    return [np.array([[1],[1]]),0.1]
    
def test_iterate_shapes_2(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    Output of the function generator
    
    """
    
    test_sdf = sdf_mask(*two_unitary_cube)
    #using function utility used
    
    assert len(utility_iterate_shapes(two_unitary_cube[0])) == 1     
    assert utility_iterate_shapes(two_unitary_cube[0])[0].size == 2     
    assert (utility_iterate_shapes(two_unitary_cube[0])[0] == True).all()     

def test_shape_as_points_2(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*two_unitary_cube)
    assert test_sdf.shape_as_points(two_unitary_cube[0]).shape[0] == 2
    assert test_sdf.shape_as_points(two_unitary_cube[0]).shape[1] == 2
    #assert test_sdf.shape_as_points(two_unitary_cube[0]) == 0


def test_generate_sides_2(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    testing length of the output
    testing value of the output
    
    """
    test_sdf = sdf_mask(*two_unitary_cube)
    assert len(utility_generate_sides(two_unitary_cube[0])) == 8
    assert utility_generate_sides(two_unitary_cube[0])[0] == ((0.5,0.5),(0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[1] == ((0.5,-0.5),(-0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[2] == ((-0.5,-0.5),(-0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[3] == ((-0.5,0.5),(0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[4] == ((1.5,0.5),(1.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[5] == ((1.5,-0.5),(0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[6] == ((0.5,-0.5),(0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[7] == ((0.5,0.5),(1.5,0.5)) 

def test_merge_cubes_2(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    

    Testing
    -------
    For the two side by side unitary cube as input, merge_cubes() returns only 
    the cordinate of the corner of the rectangle formed by the two, and the two 
    conjuction points
    """
    test_sdf = sdf_mask(*two_unitary_cube)
    assert len(test_sdf.merge_cubes(two_unitary_cube[0])) == 6
    assert test_sdf.merge_cubes(two_unitary_cube[0])[0] == (0.5,-0.5)
    assert test_sdf.merge_cubes(two_unitary_cube[0])[1] == (-0.5,-0.5)
    assert test_sdf.merge_cubes(two_unitary_cube[0])[2] == (-0.5,0.5)
    assert test_sdf.merge_cubes(two_unitary_cube[0])[3] == (0.5,0.5)
    assert test_sdf.merge_cubes(two_unitary_cube[0])[4] == (1.5,0.5)
    assert test_sdf.merge_cubes(two_unitary_cube[0])[5] == (1.5,-0.5)

def test_distance_from_poly_2(two_unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    Number of points inside the cube (calculated distance smaller than 0) or 
    along the side (calculated distance equal to 0) for different grid_finess
    
    """
    #grid_finess = 1
    test_sdf = sdf_mask(two_unitary_cube[0],1)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 2
    assert test_sdf.distances[test_sdf.distances==0].size == 0
    
    
    #grid_finess = 0.5
    test_sdf = sdf_mask(two_unitary_cube[0],0.5)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 3
    assert test_sdf.distances[test_sdf.distances==0].size == 12
    
    #grid_finess = 0.1
    test_sdf = sdf_mask(two_unitary_cube[0],0.1)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == 171
    assert test_sdf.distances[test_sdf.distances==0].size == 60
    
    #we can compare this with a theoretical value given by the grid_finess using an utility function
    # which calculates the theoretical number of points inside and along the side of the unitary cube
    test_sdf = sdf_mask(two_unitary_cube[0],0.02)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == utility_distance_from_poly_2(0.02)[0]
    assert test_sdf.distances[test_sdf.distances==0].size == utility_distance_from_poly_2(0.02)[1]


def test_sdf_2(two_unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    Value of the distance obtained
    With a grid_finess = 1 we get only 12 points in the grid.
    The corners points will have the same minimum distance from the poly,
    this is just the one fourth of the diagonal of the square defined as before
    The other grid points will have same distance, even though the two at the centre
    will be defined with a negative sign, since inside the square
    """
    
    test_sdf = sdf_mask(two_unitary_cube[0],1)
    assert len(test_sdf.sdf()[test_sdf.sdf() ==0.5]) == 6
    assert len(test_sdf.sdf()[test_sdf.sdf() ==-0.5]) == 2
    assert len(test_sdf.sdf()[abs(test_sdf.sdf()) ==np.sqrt(2)/2]) == 4
    
def test_sdf_2_1(two_unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    In the case of one figure the sdf returned is the same as the
    self.distances returned by distance_from_poly
    """
    
    test_sdf = sdf_mask(two_unitary_cube[0],1)
    assert (test_sdf.sdf()-test_sdf.distances == 0).all()

"""
--------
Unit testing with two unitary cube separated by a gap, limitating
the test only to the functions which must recognize that two 
separated figures are present
We labe the tests with 3
--------
"""
@pytest.fixture
def two_cube_separated(): 
    """

    Returns
    -------
    list
        segmentation = two_cube_separated[0] 
        grid_finess = two_cube_separated[1]
    """
    return [np.array([[1],[0],[1]]),0.1]


def test_iterate_shapes_3(two_cube_separated): 
    """
    Parameters
    ----------
    list
        segmentation = two_cube_separated[0] 
        grid_finess = two_cube_separated[1]
     
    
    Testing
    -------
    Output of the function generator gives two shapes
    
    """
    
    test_sdf = sdf_mask(*two_cube_separated)
    #using function utility used
    
    assert len(utility_iterate_shapes(two_cube_separated[0])) == 2 #two figures found, than the other functions will work regularly

def test_distance_from_poly_3(two_cube_separated): 
    """

    Parameters
    ----------
    list
        segmentation = two_cube_separated[0] 
        grid_finess = two_cube_separated[1]
    
    Testing
    -------
    We check that we have two sets of distances now, as many as the number of figures  
    """
    
    test_sdf = sdf_mask(*two_cube_separated)
    test_sdf.sdf()
    assert test_sdf.distances.shape[0]==2

def test_sdf_3(two_cube_separated): 
    """

    Parameters
    ----------
    list
        segmentation = two_cube_separated[0] 
        grid_finess = two_cube_separated[1]
    
    Testing
    -------
    Value of the distance obtained
    With a grid_finess = 1 we get only 15 points in the grid.
    The corners points of the cubes will have the same minimum distance from the two poly,
    this is just the one fourth of the diagonal of the square defined as before
    The other grid points will have same distance, even though the two at the centre
    of the two cubes will be defined with a negative sign
    """
    
    test_sdf = sdf_mask(two_cube_separated[0],1)
    assert len(test_sdf.sdf()[test_sdf.sdf() ==0.5]) == 7
    assert len(test_sdf.sdf()[test_sdf.sdf() ==-0.5]) == 2
    assert len(test_sdf.sdf()[abs(test_sdf.sdf()) ==np.sqrt(2)/2]) == 6
    
