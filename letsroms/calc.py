# Description: Fuctions for calculating derived quantities
#              from LeTSROMS model-sampled fields.
# Author/date: André Palóczy, July/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['crosstrk_flux',
           'mke',
           'eke',
           'ape',
           'pespec',
           'kespec',
           'xspec']

import numpy as np
from scipy.signal import csd
from ap_tools.utils import rot_vec
from .utils import blkwavg, strip, conform


def crosstrk_flux(Romsship, variable, kind='eddyflx', \
                  synop=False, segwise_synop=False, interp_method='linear', \
                  cache=True, xarray_out=True, verbose=True, **kw):
    """
    Calculate the cross-track mean (u_mean*Q_mean) eddy flux (u'*Q') for
    a variable "Q" sampled from ROMS output. All kwargs are
    passed to RomsShip.ship_sample().

    The mean is defined as the average of Q over each track line, such that
    u = u_mean + u' and Q = Q_mean + Q' are the total cross-track velocity
    and value of Q, respectively.

    'kind' can have the following values:

    * 'eddyflx':   Returns uQmean(z) and uQeddy(z), or the mean and eddy
                   cross-track fluxes of 'Q' for each segment.

    * 'eddytransp' Returns uQmean(segnum) and uQeddy(segnum), or the vertical
                   integral of uQmean(x, z) and uQeddy(x, z).
    """
    if not hasattr(Romsship, 'u_xtrk') and not hasattr(Romsship, 'v_atrk'):
        uship = Romsship.ship_sample('u', synop=synop, segwise_synop=segwise_synop, **kw)
        vship = Romsship.ship_sample('v', synop=synop, segwise_synop=segwise_synop, **kw)

        # Rotate velocities to along-/across-track if not already available.
        # Rotate first to east-west coordinates and then to track coordinates.
        ang_trk = Romsship.angship

    assert uship._interpm==vship._interpm
    interpm = uship._interpm
    if not hasattr(Romsship, 'ang_grd'):
        ang_grd = Romsship.angle.ravel()/Romsship._deg2rad # [radians].
        xsrad = strip(Romsship.xship)*Romsship._deg2rad
        ysrad = strip(Romsship.yship)*Romsship._deg2rad
        ang_grd =  Romsship._interpxy(ang_grd, xsrad, ysrad, interpm, 'rho')
    else:
        ang_grd = Romsship.ang_grd
    ang_tot = ang_trk - ang_grd # [degrees].

    if isinstance(variable, str): # If name of variable is given, sample it.
        shipvar = Romsship.ship_sample(variable, synop=synop, segwise_synop=segwise_synop, **kw)
    else: # If sampled variable is provided, no need to sample.
        shipvar = variable

    # Mask cells in the 'dx' array that are under the bottom.
    dx = shipvar.dx
    uship, vship, shipvar = map(strip, (uship, vship, shipvar))

    if not hasattr(Romsship, 'u_xtrk'):
        _, uship = rot_vec(uship, vship, angle=ang_tot, degrees=True)
        uship = -uship # Cross-track velocity (u) is positive to the RIGHT of the track.

    # Calculate cross-track fluxes.
    segidx = np.tile(strip(Romsship.Shiptrack.segment_index.data)[np.newaxis,:], \
                    (Romsship.N - 1, 1))[:,:-1]

    if kind=='eddyflx':
        stride = 'right'
    elif kind=='eddytransp':
        stride = 'right-up'

    uship = conform(uship, stride=stride)
    shipvar = conform(shipvar, stride=stride)
    uQmean, uQeddy = np.array([]), np.array([])
    for n in range(Romsship.Shiptrack.nsegs):
        n+=1
        fseg=segidx==n
        dxseg = dx[fseg]         # [m].
        print(dxseg)
        print(dx)
        Lseg = dxseg[0,:].sum()  # [m].
        ushipn = uship[fseg]
        shipvarn = shipvar[fseg] # Get Along-track-averaged covariance profile (n-th segment).
        # # Get Along-track-averaged mean cross-track flux profile (n-th segment).
        # uQmeann = blkwavg(ushipn, coords, dim='x')
        uQmeann = np.sum(ushipn*dxseg, axis=1)*np.sum(shipvarn*dxseg, axis=1)/Lseg**2 # [uQ].
        # # Get Along-track-averaged covariance profile (n-th segment).
        uQcovn = np.sum(ushipn*shipvarn*dxseg, axis=1)/Lseg # [uQ].
        uQeddyn = uQcovn - uQmeann                          # [uQ].

        if kind=='eddytransp':
            uQmeann = uQmeann
            uQeddyn = uQeddyn

        uQmean = np.append(uQmean, uQmean)
        uQeddy = np.append(uQeddy, uQeddyn)

    return uQmean, uQeddy


def mke(u, v, density=True):
    raise NotImplementedError


def eke(u, v, density=True):
    raise NotImplementedError


def ape(density=True):
    raise NotImplementedError


def pespec():
    """Along-track potential energy wavenumber spectrum"""
    raise NotImplementedError


def kespec():
    """Along-track kinetic energy wavenumber spectrum"""
    raise NotImplementedError


def xspec(Romsship, var1, var2, synop=False, segwise_synop=False, **kw):
    """Along-track cross-spectrum of two variables 'var1' and 'var2'."""
    assert type(var1)==type(var2), "'var1' and 'var2' must be of the same type."
    fs = Romsship.Shiptrack.sampfreq.data*Romsship._cph2hz # [Hz].
    if type(var1)==str:
        var1 = Romsship.ship_sample(var1, synop=synop, segwise_synop=segwise_synop, **kw)
        var2 = Romsship.ship_sample(var2, synop=synop, segwise_synop=segwise_synop, **kw)
    else:
        var1 = strip(var1)
        var2 = strip(var2)
    # amp, phase = coherence(a, b, **kw)
    f, xspec = csd(var1, var2, fs=fs, return_onesided=False, scaling='density')
    ampspec = np.abs(xspec)
    phasespec = np.angle(xspec)

    return f, ampspec, phasespec
