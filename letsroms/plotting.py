# Description: Plotting functions for
#              LeTSROMS classes.
# Author/date: André Palóczy, June/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['chk_synopticity']

import numpy as np
from matplotlib import pyplot as plt


def chk_synopticity(self, xn, yn, time, arr, varname, figsize=(16,8)):
    """
    Plot a comparison between a ship-sampled transect of model output
    with the instantaneous (synoptic) model transect.

    # TODO: Plot vertical lines at the times where there is model output.
    """
    if arr.ndim==2:
        fig, ax = plt.subplots(ncols=3, figsize=figsize)
        ax1, ax2, ax3 = ax
        cs = ax1.contourf(tship3_num, zship3, var3dship, 30, shading='flat')
        ax1.axis('tight')
        ax1.set_ylabel('Depth [m]')
        plt.colorbar(cs)
        cs = ax1.contourf(tship3_num, zship3, var3dship, 30, shading='flat')
        ax1.axis('tight')
        ax1.set_ylabel('Depth [m]')
        plt.colorbar(cs)
        ax = (ax1, ax2, ax3)
    elif arr.ndim==1:
        fig, ax = plt.sublots(ncols=2, figsize=figsize)
        ax1, ax2 = ax
        fts = 1
        # Get the instantaneous line at the time the ship starts the line.
        self._interpsynop(self, arrs, ti, xn, yn, interpm, pointtype)

        ax1.plot(troms, romsarr, 'b-', marker='o')
        ax1.axis('tight')
        ax1.set_ylabel('Variable [var units]')
        ax = (ax1, ax2)
    plt.show()

    return fig, ax
