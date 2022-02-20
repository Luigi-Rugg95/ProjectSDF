# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:49:37 2022

@author: Luigi
"""

#functions needed for plotting SDF

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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