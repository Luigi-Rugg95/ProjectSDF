# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:43:42 2022

@author: Luigi
"""


from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from sdf_from_binary_mask import *
from functions_for_testing import utility_distance_from_poly_1,utility_distance_from_poly_2,utility_generate_sides,utility_iterate_shapes

import numpy as np
import pytest


    
    

def test_grid_boundary():
    """
    

    Testing
    -------
    testing the grid boundary with respect to the input segmentation
    
    """
    segmentation = np.array([[1],])
    grid_finess=0.1
    
    test_sdf = sdf_mask(segmentation,grid_finess)
    assert(np.max(test_sdf.grid()[0])>=segmentation[:,0].size)
    assert(np.max(test_sdf.grid()[1])>=segmentation[0].size)
    assert(np.min(test_sdf.grid()[0])<0)
    assert(np.min(test_sdf.grid()[1])<0)
        
    
    
    
def test_grid_zero_value(): 
    """
    

    Testing
    -------
    The code raise a ZeroDivisionError when a zero value is given as grid_finess

    """
    segmentation = np.array([[1],])
    grid_finess=0   
    test_sdf = sdf_mask(segmentation,grid_finess)
    with pytest.raises(ZeroDivisionError):
           assert test_sdf.grid()


    
def test_segmentation_dimension_3D(): 
    """
    

    Testing
    -------
    testing 3D segmentation as input
    """
    segmentation=np.array([[[0,1,0],],])
    grid_finess= 0.1
    with pytest.raises(Exception):
            assert sdf_mask(segmentation,grid_finess)
    
def test_segmentation_dimension_1D(): 
    """
    

    Testing
    -------
    testing 1D segmentation as input
    """    
    
    segmentation=np.array([0,1,0])
    grid_finess= 0.1
    with pytest.raises(Exception):
            assert sdf_mask(segmentation,grid_finess)

def test_segmentation_zero_value(): 
    """
    

    Testing
    -------
    testing zero value segmentation as input
    """    

    segmentation=np.array([[0],])
    grid_finess= 0.1
    with pytest.raises(AssertionError):
            assert sdf_mask(segmentation,grid_finess)



"""
Unit testing using a unitary square centered in (0,0) as input, test are labelled with 1
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
    


def test_iterate_shapes_1_shape_found(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
     
    
    Testing
    -------
    given unitary cube as input, testing the function iterate_shapes 
    finds a single shape
    
    """
    
    assert len(utility_iterate_shapes(unitary_cube[0])) == 1     
    
def test_iterate_shapes_1_single_pixel_found(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
     
    
    Testing
    -------
    given unitary cube as input, testing the function iterate_shapes 
    returns only a pixel with bolean value True
    
    """
    
    assert (utility_iterate_shapes(unitary_cube[0])[0] == True).all()     


def test_shape_as_points_1_output_length(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a unitary cube as input testing length of the output
    """
    
    assert shape_as_points(unitary_cube[0]).shape[0] == 1
    assert shape_as_points(unitary_cube[0]).shape[1] == 2
    
def test_shape_as_points_1_output_value(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a unitary cube as input testing value of the output
    """
    
    assert shape_as_points(unitary_cube[0])[0][0]==0
    assert shape_as_points(unitary_cube[0])[0][1]==0
    

def test_generate_sides_1_length_output(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a unitary cube as input testing length of the output
    
    """
    assert len(utility_generate_sides(unitary_cube[0])) == 4
    
def test_generate_sides_1_value_output(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a unitary cube as input testing value of the output
    
    """
    assert utility_generate_sides(unitary_cube[0])[0] == ((0.5,0.5),(0.5,-0.5)) 
    assert utility_generate_sides(unitary_cube[0])[1] == ((0.5,-0.5),(-0.5,-0.5)) 
    assert utility_generate_sides(unitary_cube[0])[2] == ((-0.5,-0.5),(-0.5,0.5)) 
    assert utility_generate_sides(unitary_cube[0])[3] == ((-0.5,0.5),(0.5,0.5)) 
     

def test_merge_cubes_1_length_output(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    

    Testing
    -------
    given a unitary cube as input, the length of the output must be equal to the
    number of corners of the shape, for a unitary cube then four
    """
    
    assert len(merge_cubes(unitary_cube[0])) == 4
    
def test_merge_cubes_1_value_output(unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    

    Testing
    -------
    given a unitary cube as input testing value of the output, coordinates of the corners
    """
    assert merge_cubes(unitary_cube[0])[0] == (0.5,0.5)
    assert merge_cubes(unitary_cube[0])[1] == (0.5,-0.5)
    assert merge_cubes(unitary_cube[0])[2] == (-0.5,-0.5)
    assert merge_cubes(unitary_cube[0])[3] == (-0.5,0.5)


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
    
    grid_points = np.array([-1,-1])
    origin = np.array([0,0])
    assert(diff_point_array(grid_points,origin) == grid_points).all()
    
    grid_points = np.array([-1,1])
    assert(diff_point_array(grid_points,origin) == grid_points).all()
    
    grid_points = np.array([1,1])
    assert(diff_point_array(grid_points,origin) == grid_points).all()

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
    
    grid_points = np.array([-1,-1])
    point = np.array([1,1])
    assert(diff_point_array(grid_points,point) == [-2,-2]).all()
    
    grid_points = np.array([3,3])
    assert(diff_point_array(grid_points,point) == [2,2]).all()
    
    grid_points = np.array([0,0])
    assert(diff_point_array(grid_points,point) == [-1,-1]).all()

    
    
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


def test_calculate_distance_1_number_of_shapes(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given as input a unitary cube, we exept to obtain only one set of sdf
    """
    test_sdf = sdf_mask(*unitary_cube)
    test_sdf.sdf()
    assert test_sdf.distances.shape[0]==1
    


def test_calculate_distance_1_length_output(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    The number of points in the sdf should be equal to the number of points in the grid
    """
    test_sdf = sdf_mask(*unitary_cube)
    test_sdf.sdf()
    assert test_sdf.distances.shape[1]==test_sdf.grid()[0].shape[0]
    assert test_sdf.distances.shape[2]==test_sdf.grid()[0].shape[1]
    

def test_sdf_1_value_sdf(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Value of the distance obtained for a unitary cube as input
    With a grid_finess = 1 we get only 9 points in the grid.
    The corners points will have the same minimum distance from the poly,
    this is just one fourth of the diagonal of the square defined by the grid
    The other grid points will have same distance, even though the one at the centre
    will be defined with a negative sign, since inside the square
    """
    
    test_sdf = sdf_mask(unitary_cube[0],1)
    assert len(test_sdf.sdf()[test_sdf.sdf() ==0.5]) == 4
    assert len(test_sdf.sdf()[test_sdf.sdf() ==-0.5]) == 1
    assert len(test_sdf.sdf()[abs(test_sdf.sdf()) ==np.sqrt(2)/2]) == 4
    

def test_sdf_1_output_comparison_with_calculate_distance(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    In the case of one figure the sdf returned is the same as the
    one obtained by the funcition calculate_distance
    """
    
    test_sdf = sdf_mask(unitary_cube[0],1)
    assert (test_sdf.sdf()-test_sdf.distances == 0).all()


"""
Unit testing using two unitary squares side by side, test labelled with 2
"""




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
    
