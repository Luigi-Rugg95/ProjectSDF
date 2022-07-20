# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:43:42 2022

@author: Luigi
"""


from sdf_from_binary_mask import sdf_from_binary_mask as sdf_mask
from sdf_from_binary_mask import *
from functions_for_testing import utility_distance_from_poly_1,utility_generate_sides,utility_iterate_shapes

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
    given unitary cube as input, testing that the function iterate_shapes 
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
    given unitary cube as input, testing that the function iterate_shapes 
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


def test_diff_point_array_1_length_output(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a grid finess equal to one (9 total points )and a unitary cube (4 vertices) 
    as input, the difference between each point of the grid and 
    each vertex should return an array with 9 points, each of it with 4 2D vector coordinates
    """    
    
    grid_points = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    vertices = np.array([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]])
    assert diff_point_array(grid_points,vertices).shape[0] == 9
    assert diff_point_array(grid_points,vertices).shape[1] == 4
    assert diff_point_array(grid_points,vertices).shape[2] == 2
    
    
def test_diff_point_array_1_value_output(unitary_cube): 
    """
    

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    given a grid finess equal to one a unitary cube, we can compare the 
    difference between each grid point and each vertex with the output of
    diff_point_array
    """    
    
    grid_points = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    vertices = np.array([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]])
    assert (diff_point_array(grid_points,vertices)[0][0] == grid_points[0]-vertices[0]).all
    assert (diff_point_array(grid_points,vertices)[0][1] == grid_points[0]-vertices[1]).all
    assert (diff_point_array(grid_points,vertices)[0][2] == grid_points[0]-vertices[2]).all
    assert (diff_point_array(grid_points,vertices)[0][3] == grid_points[0]-vertices[3]).all
    
    assert (diff_point_array(grid_points,vertices)[1][0] == grid_points[1]-vertices[0]).all
    assert (diff_point_array(grid_points,vertices)[1][1] == grid_points[1]-vertices[1]).all
    assert (diff_point_array(grid_points,vertices)[1][2] == grid_points[1]-vertices[2]).all
    assert (diff_point_array(grid_points,vertices)[1][3] == grid_points[1]-vertices[3]).all
    
    assert (diff_point_array(grid_points,vertices)[2][0] == grid_points[2]-vertices[0]).all
    assert (diff_point_array(grid_points,vertices)[2][1] == grid_points[2]-vertices[1]).all
    assert (diff_point_array(grid_points,vertices)[2][2] == grid_points[2]-vertices[2]).all
    assert (diff_point_array(grid_points,vertices)[2][3] == grid_points[2]-vertices[3]).all
    
    assert (diff_point_array(grid_points,vertices)[3][0] == grid_points[3]-vertices[0]).all
    assert (diff_point_array(grid_points,vertices)[3][1] == grid_points[3]-vertices[1]).all
    assert (diff_point_array(grid_points,vertices)[3][2] == grid_points[3]-vertices[2]).all
    assert (diff_point_array(grid_points,vertices)[3][3] == grid_points[3]-vertices[3]).all
    
    
def test_distance_from_poly_1_points_inside_the_shape(unitary_cube): 
    """

    Parameters
    ----------
    list
        segmentation = unitary_cube[0] 
        grid_finess = unitary_cube[1]
    
    Testing
    -------
    Given as input a unitary cube, considering the cruteria used to define the grid,
    we can find a general expression for the number of points which lays inside the shape
    as function of the grid_finess: 
        with a unitary cube the grid is a square 2x2, with number of points = ((2/grid_finess)+1)^2
    The cube is centered in the origin and the vertices coordinates are whether 0.5 or -0.5, 
    From this we can use an algebraic calculation in order to predict the number of points 
    inside the unitary cube considering the variation of the grid_finess
    """
    test_sdf = sdf_mask(unitary_cube[0],0.02)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == utility_distance_from_poly_1(0.02)
    
    test_sdf = sdf_mask(unitary_cube[0],0.03)
    test_sdf.sdf()
    assert test_sdf.distances[test_sdf.distances<0].size == utility_distance_from_poly_1(0.03)
    


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
    
def test_iterate_shapes_2_shape_found(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing that the function iterate_shapes 
    finds a single shape
    
    """
    

    assert len(utility_iterate_shapes(two_unitary_cube[0])) == 1     
    assert utility_iterate_shapes(two_unitary_cube[0])[0].size == 2     
    assert (utility_iterate_shapes(two_unitary_cube[0])[0] == True).all()     


def test_iterate_shapes_2_two_pixel_found(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing that the function iterate_shapes 
    finds two pixels
    
    """
    
    assert utility_iterate_shapes(two_unitary_cube[0])[0].size == 2     
    assert (utility_iterate_shapes(two_unitary_cube[0])[0] == True).all()     



def test_shape_as_points_2_length_output(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing length of the output
    
    """
    
    assert shape_as_points(two_unitary_cube[0]).shape[0] == 2
    assert shape_as_points(two_unitary_cube[0]).shape[1] == 2

def test_shape_as_points_2_value_output(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing value of the output
    
    """
    
    assert shape_as_points(two_unitary_cube[0])[0][0]==0
    assert shape_as_points(two_unitary_cube[0])[0][1]==0
    assert shape_as_points(two_unitary_cube[0])[1][0]==1
    assert shape_as_points(two_unitary_cube[0])[1][1]==0
    

def test_generate_sides_2_length_output(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing length of the output
    
    """
    assert len(utility_generate_sides(two_unitary_cube[0])) == 8
    
def test_generate_sides_2_value_output(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    
    Testing
    -------
    given two unitary cubes side by side as input, testing value of the output
    
    """
    assert utility_generate_sides(two_unitary_cube[0])[0] == ((0.5,0.5),(0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[1] == ((0.5,-0.5),(-0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[2] == ((-0.5,-0.5),(-0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[3] == ((-0.5,0.5),(0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[4] == ((1.5,0.5),(1.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[5] == ((1.5,-0.5),(0.5,-0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[6] == ((0.5,-0.5),(0.5,0.5)) 
    assert utility_generate_sides(two_unitary_cube[0])[7] == ((0.5,0.5),(1.5,0.5)) 


def test_merge_cubes_2_length_output(two_unitary_cube): 
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
    conjuction points, thus six points
    """
    assert len(merge_cubes(two_unitary_cube[0])) == 6


def test_merge_cubes_2_length_output(two_unitary_cube): 
    """
    Parameters
    ----------
    list
        segmentation = two_unitary_cube[0] 
        grid_finess = two_unitary_cube[1]
    

    Testing
    -------
    given two unitary cubes side by side as input, testing value of the output
    
    """
    
    assert merge_cubes(two_unitary_cube[0])[0] == (0.5,-0.5)
    assert merge_cubes(two_unitary_cube[0])[1] == (-0.5,-0.5)
    assert merge_cubes(two_unitary_cube[0])[2] == (-0.5,0.5)
    assert merge_cubes(two_unitary_cube[0])[3] == (0.5,0.5)
    assert merge_cubes(two_unitary_cube[0])[4] == (1.5,0.5)
    assert merge_cubes(two_unitary_cube[0])[5] == (1.5,-0.5)



def test_sdf_2_output_sdf(two_unitary_cube): 
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


def test_iterate_shapes_3_shape_found(two_cube_separated): 
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
    
    assert len(utility_iterate_shapes(two_cube_separated[0])) == 2 #two figures found, than the other functions will work regularly

def test_distance_from_poly_3_length_output(two_cube_separated): 
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

def test_sdf_3_output_sdf(two_cube_separated): 
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
    
