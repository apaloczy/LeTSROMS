# Description: Plotting functions for
#              LeTSROMS classes.
# Author/date: André Palóczy, June/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['chk_synopticity']

import numpy as np
from matplotlib import pyplot as plt


def chk_synopticity(varship, varsynop, ship_speed, sampling_period, contour_levels=50, logscale=False):
    """
    TODO: Add faint vertical lines indicating sampled profiles and
    the separations between occupations.

    Plot a comparison between a ship-sampled transect of model output
    with the instantaneous (synoptic) model transect.

    # TODO: Plot vertical lines at the times where there is model output.
    """
    isxr = varship.fromDataArray
    assert varship.ndim==varsynop.ndim, "Synoptic and ship-sampled variables dimensions mismatch."
    assert varship.name==varsynop.name, "Synoptic and ship-sampled fields are not the same variable."
    varname = varship.name
    varunits = varship.units
    shipvar = varship.vship
    synopvar = varsynop.vship
    diffvar = synopvar - shipvar
    x = varship.dship
    if varship.ndim==2:
        x1 = x.copy()
        z = varship.zship
        if logscale:
            shipvar = np.log10(shipvar)
            synopvar = np.log10(synopvar)
            diffvar = synopvar - shipvar
        dz = z[1:,:] - z[:-1,:]
        diffvar_bar = 0.5*(diffvar[1:,:] + diffvar[:-1,:])*dz
        diffvar_bar = diffvar_bar.sum(axis=0)
        diff_hi = np.max(np.abs(diffvar))
        diff_lo = - diff_hi
        fig, ax = plt.subplots(nrows=4, sharex=True)
        ax1, ax2, ax3, ax4 = ax

        x, z = [np.array(arr) for arr in np.broadcast_arrays(x, z)]
        cs1 = ax1.contourf(x, z, shipvar, contour_levels, shading='flat')
        cs2 = ax2.contourf(x, z, synopvar, contour_levels, shading='flat')
        cs3 = ax3.contourf(x, z, diffvar, contour_levels, vmin=diff_lo, vmax=diff_hi, cmap=plt.cm.seismic, shading='flat')
        ax4.plot(x, diffvar_bar, 'k', marker='o', ms=3, label=r'Depth-avg (synop - ship)')
        ax4.axis('tight')
        ax4.grid()
        ax4.axhline(y=0, linewidth=2.0, color='r', alpha=0.5)
        ax2.set_ylabel(r'Depth [m]', fontsize=18, fontweight='black')
        ax4.set_ylabel(r'Synop - ship', fontsize=15, fontweight='black')
        ax4.set_xlabel(r'Along-track distance [km]', fontsize=18, fontweight='black')
        ax1.set_title(r'Ship-sampled (%s) - every %d min, at %.1f kn'%(varname, sampling_period, ship_speed), fontsize=15, fontweight='black')
        ax2.set_title(r'Synoptic (%s)'%varname, fontsize=14, fontweight='black')
        ax3.set_title(r'Synoptic - ship-sampled (%s)'%varname, fontsize=14, fontweight='black')
        ax4.set_title(r'Depth-averaged synoptic - ship-sampled (%s)'%varname, fontsize=14, fontweight='black')
        plt.colorbar(cs2, ax=(ax1, ax2), use_gridspec=True)
        plt.colorbar(cs3, ax=(ax3, ax4), use_gridspec=True)
        ax = (ax1, ax2, ax3, ax4)
    elif varship.ndim==1:
        fig, ax = plt.subplots(sharex=True, nrows=2)
        ax1, ax2 = ax
        ax1.plot(x, shipvar, 'grey', marker='o', ms=3, label='ship')
        ax1.plot(x, synopvar, 'k', marker='o', ms=3, label='synoptic')
        ax2.plot(x, diffvar, 'k', marker='o', ms=3)
        ax1.legend(loc='best')
        ax1.set_xlim(x[0], x[-1])
        ax2.axhline(y=0, linewidth=2.0, color='r', alpha=0.5)
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel(r'%s [%s]'%(varname, varunits), fontsize=18, fontweight='black')
        ax2.set_ylabel(r'Synop - ship', fontsize=18, fontweight='black')
        ax2.set_xlabel(r'Along-track distance [km]', fontsize=18, fontweight='black')
        ax1.set_title(r'Ship-sampled (%s) - every %d min, at %.1f kn'%(varname, sampling_period, ship_speed), fontsize=15, fontweight='black')
        ax2.set_title(r'Synoptic - ship-sampled (%s)'%varname, fontsize=14, fontweight='black')
        ax = (ax1, ax2)
    plt.show()

    return fig, ax
