# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:49:37 2022

@author: Luigi
"""

#functions needed for plotting SDF

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import configparser #In order to read the data
import numpy as np


# color map for plotting SDF
δ = 0.001
mycmap = LinearSegmentedColormap.from_list(
    "mycmap",
    colors=[
         (0, "purple"), 
         (0.25, "red"), 
         (0.5-δ, "orange"), 
         (0.5+δ, "green"), 
         (1, "cyan"),
    ], 
    N=32,
)



def plotting_shape(distances,edges_xn,edges_yn,gradient_norm, X,Y, poly): 
    norm_d = mcolors.TwoSlopeNorm(vmin=distances.min(), vcenter=0, vmax=distances.max())
    norm_gx = mcolors.TwoSlopeNorm(vmin=edges_xn.min(), vcenter=0, vmax=edges_xn.max())
    norm_gy = mcolors.TwoSlopeNorm(vmin=edges_yn.min(), vcenter=0, vmax=edges_yn.max())
    props = dict(aspect="equal", origin="lower", extent=(X.min(), X.max(), Y.min(), Y.max()))

    gy = r"$\nabla {SDF}_y$"
    gx = r"$\nabla {SDF}_x$"
    gn = r"$|| \nabla SDF ||$"
    fig, axes = plt.subplot_mosaic([["SDF", gn], [gy, gx]], figsize=(16, 12))
    for name, ax in axes.items():
        ax.set_title(name, fontsize=20, fontfamily="serif", y=1.025)
    
    axes["SDF"].imshow(distances.T, norm=norm_d, cmap=mycmap, **props) #cos'è .T????
    axes[gn].imshow(gradient_norm.T, **props)
    axes[gy].imshow(edges_yn.T, norm=norm_gx, cmap='PiYG', **props)
    axes[gx].imshow(edges_xn.T, norm=norm_gy, cmap='PiYG', **props)
    
    for ax in axes.values():
        polygons = [Polygon(poly, True)]
        poly_collection = PatchCollection(polygons, alpha=0.5, color="gray")
        ax.add_collection(poly_collection)
        
    plt.show()
    return