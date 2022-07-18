# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 10:30:50 2022

@author: Luigi
"""

import numpy as np
from scipy.ndimage import label
import os
import pylab as plt

#import pylab as plt



def load_segmentation():
    """
    

    Returns
    -------
    segmentations : list of numpy.ndarray
        All inputs are loaded and appended to the lsit
    files : list of string
        file names useful to label the output

    """
    segmentations = []
    for root, dirs, files in os.walk('./input/'):
        for file in files:
            segmentations.append((plt.imread('./input/' + file)>0).astype(np.uint32))
    return segmentations,files


def iterate_shapes(image):
        """
        

        Parameters
        ----------
        image : numpy.ndarray 
            input binary blob mask given in the main (segmentation in the class)

        Yield 
        -----
            generator function conditional statement for 
            labelled figures and number of labels
            
        Description
        -----------    
            it finds disconnected shapes in a black and white 
            image and it returns them one by one
        
        """
        
        labeled_array, num_features = label(image)
        for idx in range(1, num_features+1):
            yield labeled_array==idx
    


class sdf_from_binary_mask: 
    
    def __init__(self,segmentation,grid_finess):
        """

        Parameters
        ----------
        segmentation : numpy.ndarray
            binary mask given as initial input for calculating the sdf
        grid_finess : float
            finess of the grid used to calculate the sdf

        Returns
        -------
        None.

        """
        
        if np.size(segmentation.shape)!=2: 
            raise Exception("Wrong dimensions of the input")
        
        assert len(segmentation[segmentation!=0])!=0, "No segmentation found"
    
    
        self.segmentation = segmentation
        self.grid_finess = abs(grid_finess)
        self.distances = np.array([[]])
        
       
    def grid(self): 
        """
    
        Returns
        -------
        X: numpy.ndarray
        Y: numpy.ndarray
            fine grid used to calculate the distances, the limit of the 
            grid are set in order to contain the whole segmentation
        
        Example
        -------
        1. An input of the form segmentation = segmentation = np.array([[1],])
        which is a unitary square centered in the origin,
        will return a grid with boundary [-1,+1] in both x and y

        2. An input of the form segmentation = segmentation = np.array([[1,0],)
        which is a unitary square centered in the origin,
        will return a grid with boundary [-1,+2] in x and boundary [-1,+1] in y

        """
        limit_grid = [self.segmentation[:,0].size, self.segmentation[0].size]
        #creating a meshgrid
        X, Y = np.mgrid[-1:limit_grid[0]+self.grid_finess:self.grid_finess,-1:limit_grid[1]+self.grid_finess:self.grid_finess]
        
        
        return X,Y    
    
    """
    --------------------------------------------
    Transforming a binary blob mask into a shape
    --------------------------------------------
    
    """
    
    
    def shape_as_points(self,shape):
        """
        

        Parameters
        ----------
        shape : numpy.ndarray type bool or int of 0,1
            shape returned by iterate_shapes() function generator for
            the input segmentation

        Returns
        -------
        points : numpy.ndarray shape (2,:)
            coordinates of the center of each pixels
        
        Description
        -------
            given a binary mask blob it returns all the ponts contained
            within it, it returns basically the center of each pixel
        """
        shape = shape.astype(bool)
        X, Y = np.mgrid[:shape.shape[0], :shape.shape[1]]
        X = X[shape]
        Y = Y[shape]
        points = np.stack([X, Y]).T
        
        return points
    
    
    def generate_sides(self,poly, d=0.5):
        """
        

        Parameters
        ----------
        poly : numpay.ndarray shape (2,:)
            coordinates of the center each pixel of the binary mask
        d : float
            The default is 0.5, wich means that it defines a side length of 
            one pixel
        
        Yields     
        ------    
        gives the generator function for the square of unitary length
        surrounding each pixel
        
        Description
        ------
            from the centre of each pixel given in the binary mask 
            it returns a square whose centre is the given pixel  
        
        """
        assert poly.shape[1]==2 #needed for 2D sdf
        for x, y in poly:
            yield (x+d, y+d), (x+d, y-d)
            yield (x+d, y-d), (x-d, y-d)
            yield (x-d, y-d), (x-d, y+d)
            yield (x-d, y+d), (x+d, y+d)
    
    def merge_cubes(self,shape):
        """
        

        Parameters
        ----------
        shape : numpy.ndarray type bool 
            shape returned by iterate_shapes() function generator for
            the input segmentation

        Returns
        -------
        final_points : numpay.ndarray
            final coordinates of the polygon 

        Description
        -------
            from the squares created in generate_sides,
            this function will merge all the cube in a single shape
        
        """
        
        points = self.shape_as_points(shape)
        sides_duplicated = {s for s in self.generate_sides(points)}
        # the sides that are duplicated are inside the shape and needs to be removed
        sides = {(p1, p2) for p1, p2 in sides_duplicated if (p2, p1) not in sides_duplicated}
        # algorithm to re-thread the sides in a polygon
        final_points = []
        pa, pb = next(iter(sides))
        ended = False
        while not ended:
            for p1, p2 in sides:
                if p1==pb:
                    final_points.append(p1)
                    pa, pb = p1, p2
                    if p2 == final_points[0]:
                        ended=True
                    break
        return final_points
    
    
    """
    -------------------------
    Calculating the distances
    -------------------------
    
    """
    
    def diff_point_array(self, A, B):
        """
        

        Parameters
        ----------
        A: numpy.ndarray
            contiguous array of the grid points
        B: numpy.ndarray
            contiguous array of the coordinates of the polygon vertex
        
        Returns
        -------
        C : numpy.ndarray
            it returns all the differences as vector between each point
            of the grid and each vertex
        
        """ 
        #print(A)
        #print(B)
        assert A.shape[-1] == B.shape[-1]
        A_p = A.reshape(*A.shape[:-1], *np.ones_like(B.shape[:-1]), A.shape[-1])
        B_p = B.reshape(*np.ones_like(A.shape[:-1]), *B.shape)
        C = A_p - B_p 
        assert C.shape == (*A.shape[:-1], *B.shape[:-1], A.shape[-1])
        return C

    
    
    
    def distance_from_poly(self,poly,points):
        """

        Parameters
        ----------
        poly : list 
            polygon obtained from the input binary mask    
            Each element has length 2, given that 
            we are working in a 2D environment.
            
        points : numpy.ndarray shape (2,:)
            coordinates of the points of the grid

        Returns
        -------
        sdf : numpy.ndarray 
            shape equal to the total number of points in the grid
            calculated sdf
        
        Description
        -----------
        the function is the one which calculates effectively the minimum distance
        beetween one point of the grid and the side of the shape (poly here), and
        this is iterated for all the points of the grid. The result will be positive
        if the point lies outside the shape, and negative if the point lies 
        inside the shape. 

        """
        p = np.ascontiguousarray(points)
        # generate the list of points (the actual list, and the list of the next point in the poly)
        vi = np.array(poly)
        vj = np.r_[vi[1:], vi[:1]]
        # difference (as vector) between each vertex and the following one
        e = vj - vi
        # difference (as vector) between each point and each vertex
        w = self.diff_point_array(p, vi)
        # calculate the distance from each segment
        ee = np.einsum("ij, ij -> i", e, e) # scalar product keeping the right sizes
        we = np.einsum("kij, ij -> ki", w, e) # scalar product keeping the right sizes
        b = w - e*np.clip( we/ee , 0, 1)[..., np.newaxis]
        bb = np.einsum("kij, kij -> ki", b, b) # scalar product keeping the right sizes
        # the distance is the minimum of the distances from all the points
        d = np.sqrt(np.min(bb, axis=-1))
        # check if the point is inside or outside
        c1 = p[:, np.newaxis, 1]>=vi[np.newaxis, :, 1]
        c2 = p[:, np.newaxis, 1]<vj[np.newaxis, :, 1]
        c3 = e[..., 0]*w[..., 1]>e[..., 1]*w[..., 0]
        c = np.stack([c1, c2, c3])
        cb = c.all(axis=0) | ((~c).all(axis=0))
        cs = np.where(cb, -1, 1)
        s = np.multiply.reduce(cs.T)
        sdf = s*d
        return sdf
    
    
    def calculate_distances(self):         
        """
    
        Returns
        -------
        Appends the calculated distances to the init self.distances list
        
        Description
        -----------
        the process is here iterated: 
            1. Shapes are found and labelled
            2. The unitary cubes which forms a shape are merged and the side are returned
            3. The SDF for each shape is calculated and stored in the list distances
        """
        X,Y = self.grid()
        
        XY = np.dstack([X, Y])
        points_to_sample = XY.reshape(-1, 2)
        
        
        for shape in iterate_shapes(self.segmentation):
            polygon = self.merge_cubes(shape)
            distance= self.distance_from_poly(polygon, points_to_sample)
            distance = distance.reshape(*XY.shape[:-1])
            if self.distances.size == 0: 
                self.distances = np.array([distance])
            else:self.distances = np.append(self.distances,[distance],axis=0)
        return 
        
    def sdf(self):
        """
    
        Returns
        -------
        final_distance : numpy.float64
            calculated SDF referred to the merge of the shape

        Description
        -----------
        Starting from all the calculated distances for each shape found, 
        this will find the minimum distances and merge them to obtain the final result
        which can be plotted using plotting() in plotting.py
        """
        
        #getting a list of distance whose length will be the number of shapes
        self.calculate_distances()
        
        #finding the minimum distance between different points and the shapes, in the case of one figure it returns self.distances
        final_distance = np.ones_like(self.distances.shape[0])*float("inf")
        for dist_matrix in self.distances:
            final_distance = np.minimum(final_distance, dist_matrix)
        
        self.write_sdf(final_distance)
        
        return final_distance
    
    def write_sdf(self,sdf_data): 
        """
        

        Parameters
        ----------
        sdf_data : numpy.ndarray
            calculated sdf

        Description
        -------
        Write the sdf and the segmentation respectively in shape.txt an input.txt
        The data can be loaded using the function numpy.loadtxt in a numpy.array
        which can be used to plot the values
        """
        first_line = "SDF data obtained using a grid finess =" + str(self.grid_finess) 
        np.savetxt("sdf_output.txt",sdf_data, header = first_line)
        np.savetxt("input.txt",self.segmentation, header = "segmantation")
        
        return
    
    