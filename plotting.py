# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:49:37 2022

@author: Luigi
"""

#functions needed for plotting SDF

#from matplotlib.patches import Circle, Wedge, Polygon
#from matplotlib.collections import PatchCollection
#import seaborn as sns

import pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# color map for plotting SDF
delta = 0.001
mycmap = LinearSegmentedColormap.from_list(
    "mycmap",
    colors=[
         (0, "purple"), 
         (0.25, "red"), 
         (0.5-delta, "orange"), 
         (0.5+delta, "green"), 
         (1, "cyan"),
    ], 
    N=32,
)


def plotting(final_distance, X, Y):
    props = dict(aspect="equal", origin="lower", extent=(X.min(), X.max(), Y.min(), Y.max()))
    fig, ax = plt.subplots(figsize=(np.minimum(np.max(X),20), np.minimum(np.max(Y),20)), dpi=300)
    norm_d = mcolors.TwoSlopeNorm(vmin=final_distance.min(), vcenter=0, vmax=final_distance.max())
    ax.imshow(final_distance.T, norm=norm_d, cmap=mycmap, **props)
    fig.savefig('shape.png')
    #plt.close('all')
    return

