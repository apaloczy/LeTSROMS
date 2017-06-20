# Description: Plotting functions for
#              LeTSROMS classes.
# Author/date: André Palóczy, June/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['chk_synopticity']

import numpy as np
from matplotlib import pyplot as plt


def chk_synopticity(varship, varsynop, nclevs=50):
    """
    TODO: Add faint vertical lines indicating sampled profiles and
    the separations between occupations.

    Plot a comparison between a ship-sampled transect of model output
    with the instantaneous (synoptic) model transect.

    # TODO: Plot vertical lines at the times where there is model output.
    """
    assert varship.ndim==varsynop.ndim, "Synoptic and ship-sampled variables dimensions mismatch."
    x = varship.ship_dist
    if varship.ndim==2:
        z = varship.ship_depth
        shipvar = varship.values
        synopvar = varsynop.values
        diffvar = synopvar - shipvar
        diff_hi = np.max(np.abs(diffvar))
        diff_lo = - diff_hi
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
        ax1, ax2, ax3 = ax
        cs1 = ax1.contourf(x, z, shipvar, nclevs, shading='flat')
        cs2 = ax2.contourf(x, z, synopvar, nclevs, shading='flat')
        cs3 = ax3.contourf(x, z, diffvar, nclevs, vmin=diff_lo, vmax=diff_hi, cmap=plt.cm.seismic, shading='flat')
        ax2.set_ylabel('Depth [m]')
        ax3.set_xlabel('Along-track distance [km]')
        plt.colorbar(cs2, ax=(ax1, ax2), use_gridspec=True)
        plt.colorbar(cs3, ax=ax3, use_gridspec=True)
        # plt.colorbar(cs)
        ax = (ax1, ax2, ax3)
    elif varship.ndim==1:
        x = varship.ship_dist
        shipvar = varship.values
        synopvar = varsynop.values
        diffvar = synopvar - shipvar
        fig, ax = plt.subplots(sharex=True, nrows=2)
        ax1, ax2 = ax
        ax1.plot(x, shipvar, 'grey', marker='o', ms=3, label='ship')
        ax1.plot(x, synopvar, 'k', marker='o', ms=3, label='synoptic')
        ax2.plot(x, diffvar, 'k', marker='o', ms=3, label='synoptic - ship')
        ax2.axhline(y=0, linewidth=2.0, color='r', alpha=0.5)
        ax1.grid()
        ax2.grid()
        # ax2.set_ylabel('Var [var units]')
        ax2.set_xlabel('Along-track distance [km]')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax = (ax1, ax2)
    plt.show()

    return fig, ax
