# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:02:01 2022

@author: Luigi
"""

import numpy as np
from scipy.ndimage import label, generate_binary_structure

def diff_point_array(A, B):
        assert A.shape[-1] == B.shape[-1]
        A_p = A.reshape(*A.shape[:-1], *np.ones_like(B.shape[:-1]), A.shape[-1])
        B_p = B.reshape(*np.ones_like(A.shape[:-1]), *B.shape)
        C = A_p - B_p 
        assert C.shape == (*A.shape[:-1], *B.shape[:-1], A.shape[-1])
        return C

    # dato un poligono descritto come lista di punti ed una matrice di posizioni x, y, ne calcola la sdf
def distance_from_poly(poly, points):
        p = np.ascontiguousarray(points)
        # generate the list of points (the actual list, and the list of the next point in the poly)
        vi = np.array(poly)
        vj = np.r_[vi[1:], vi[:1]]
        # difference (as vector) between each vertex and the following one
        e = vj - vi
        # difference (as vector) between each point and each vertex
        w = diff_point_array(p, vi)
        # calculate the distance from each segment
        ee = np.einsum("ij, ij -> i", e, e) # scalar product keeping the righ sizes
        we = np.einsum("kij, ij -> ki", w, e) # scalar product keeping the righ sizes
        b = w - e*np.clip( we/ee , 0, 1)[..., np.newaxis]
        bb = np.einsum("kij, kij -> ki", b, b) # scalar product keeping the righ sizes
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
    
    # calcola il gradiente numerico della sdf, sia in x, y e come norma
def sdf_complex_gradient(sdf):
        edges_yn, edges_xn = np.gradient(sdf, edge_order=1)
        gradient_norm = np.sqrt(edges_yn**2 + edges_xn**2)
        return edges_xn, edges_yn, gradient_norm