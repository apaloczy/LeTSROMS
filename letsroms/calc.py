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


def crosstrk_flux(Romsship, variable, kind='eddyflux', normalize=True, \
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

    * 'eddyflux':   Returns uQmean(z) and uQeddy(z), or the mean and eddy
                   cross-track fluxes of 'Q' for each segment.

    * 'eddytransp' Returns uQmean(segnum) and uQeddy(segnum), or the vertical
                   integral of uQmean(x, z) and uQeddy(x, z).
    """
    if hasattr(Romsship, '_FIXED_DX'):
        fix_dx = True
    else:
        fix_dx = False

    # Rotate velocities to along-/across-track if not already available. Rotate
    # first from grid coordinates to east-west coordinates and then to track \
    # coordinates.
    ang_tot = Romsship.angship - Romsship.anggrdship # [degrees].
    Uship = Romsship.ship_sample('u', synop=synop, \
                                 segwise_synop=segwise_synop, \
                                 fix_dx=fix_dx, **kw)
    Vship = Romsship.ship_sample('v', synop=synop, \
                                 segwise_synop=segwise_synop, \
                                 fix_dx=fix_dx, **kw)

    Uship = strip(Uship)
    Vship = strip(Vship)
    vship, uship = rot_vec(Uship, Vship, angle=ang_tot, degrees=True)
    uship = -uship # Cross-track velocity (u) is positive to the RIGHT of the track.

    if variable=='u':
        shipvar = uship.copy()
    elif variable=='v':
        shipvar = vship.copy()
    else:
        shipvar = Romsship.ship_sample(variable, synop=synop, \
                                       segwise_synop=segwise_synop, \
                                       fix_dx=fix_dx, **kw)

    # Mask cells in the 'dx' array that are under the bottom.
    dx = shipvar.dx
    dzm = shipvar.dzm # 'dz' at the points in between ship samples.
    shipvar = strip(shipvar)

    Nm = Romsship.N - 1

    # Calculate cross-track fluxes.
    if fix_dx:
        stride = 'up'
        stride_dxdz = None
        occidx = Romsship.Shiptrack.occupation_indexm.data
        segidx = Romsship.Shiptrack.segment_indexm.data
    else:
        stride = 'right-up'
        stride_dxdz = 'right'
        occidx = Romsship.Shiptrack.occupation_index.data
        segidx = Romsship.Shiptrack.segment_index.data
    Lsegs = Romsship.Shiptrack.seg_lengths.data/Romsship._m2km # [m].
    for m in range(Romsship.Shiptrack.nrepeat):
        for n in range(Romsship.Shiptrack.nsegs):
            fseg=np.where(np.logical_and(occidx==m+1, segidx==n+1))[0]
            fsegl, fsegr = fseg[0], fseg[-1]
            dxseg = conform(dx[:, fsegl:fsegr], stride=stride_dxdz)
            dzseg = conform(dzm[:, fsegl:fsegr], stride=stride_dxdz)
            ushipn = conform(uship[:, fsegl:fsegr], stride=stride)
            shipvarn = conform(shipvar[:, fsegl:fsegr], stride=stride)
            Lseg = Lsegs[n]
            # uQmeann = blkwavg(ushipn, coords, dim='x') # FIXME
            # Get along-track-averaged mean cross-track profile and covariance (for this segment).
            uQmeann = np.sum(ushipn*dxseg, axis=1)*np.sum(shipvarn*dxseg, axis=1)/Lseg**2
            uQcovn = np.sum(ushipn*shipvarn*dxseg, axis=1)/Lseg
            uQeddyn = uQcovn - uQmeann # uQ = f(z).
            if kind=='eddytransp':     # uQ = f(segment #).
                dzsegavg = dzseg.mean(axis=1)
                uQmeann = np.sum(uQmeann*dzsegavg) # [u]*[Q]*m, transports per
                uQeddyn = np.sum(uQeddyn*dzsegavg) # unit along-track length.
                if normalize:
                    havg = dzsegavg.sum()
                    uQmeann = uQmeann/havg # [u]*[Q], along-track
                    uQeddyn = uQeddyn/havg # averaged Q transports.
                if m==n==0:
                    uQmean = uQmeann
                    uQeddy = uQeddyn
                else:
                    uQmean = np.append(uQmean, uQmeann)
                    uQeddy = np.append(uQeddy, uQeddyn)
            if kind=='eddyflux':
                if m==n==0:
                    uQmean = uQmeann[:,np.newaxis] # [u]*[Q], along-track
                    uQeddy = uQeddyn[:,np.newaxis] # averaged Q transports.
                else:
                    uQmean = np.hstack((uQmean, uQmeann[:,np.newaxis]))
                    uQeddy = np.hstack((uQeddy, uQeddyn[:,np.newaxis]))

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
