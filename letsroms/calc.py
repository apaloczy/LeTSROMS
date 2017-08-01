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


def crosstrk_flux(Romsship, variable, kind='eddyflux', \
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
    # if not hasattr(Romsship, 'u_xtrk') and not hasattr(Romsship, 'v_atrk'):
    uship = Romsship.ship_sample('u', synop=synop, segwise_synop=segwise_synop)
    vship = Romsship.ship_sample('v', synop=synop, segwise_synop=segwise_synop)

    # Rotate velocities to along-/across-track if not already available. Rotate
    # first from grid coordinates to east-west coordinates and then to track coordinates.
    ang_tot = Romsship.angship - Romsship.anggrdship # [degrees].

    # if isinstance(variable, str): # If a name of a variable is given, sample it.
    shipvar = Romsship.ship_sample(variable, synop=synop, segwise_synop=segwise_synop, **kw)
    # else: # No need to sample if wanted variable is already provided.
    #     shipvar = variable

    # Mask cells in the 'dx' array that are under the bottom.
    dx = shipvar.dx
    dzm = shipvar.dzm # 'dz' at the points in between ship samples.
    uship, vship, shipvar = map(strip, (uship, vship, shipvar))

    # if not hasattr(Romsship, 'u_xtrk'):
    _, uship = rot_vec(uship, vship, angle=ang_tot, degrees=True)
    uship = -uship # Cross-track velocity (u) is positive to the RIGHT of the track.

    Nm = Romsship.N - 1
    occidx = strip(Romsship.Shiptrack.occupation_indexm.data)
    segidx = strip(Romsship.Shiptrack.segment_indexm.data)

    # Calculate cross-track fluxes.
    stride = 'right-up'
    Lsegs = Romsship.Shiptrack.seg_lengths.data/Romsship._m2km # [m].
    for m in range(Romsship.Shiptrack.nrepeat):
        for n in range(Romsship.Shiptrack.nsegs):
            fseg=np.where(np.logical_and(occidx==m+1, segidx==n+1))[0]
            fsegl, fsegr = fseg[0], fseg[-1]
            dxseg = dx[:, fsegl:fsegr]
            dzseg = dzm[:, fsegl:fsegr]
            ushipn = conform(uship[:, fsegl:fsegr], stride=stride)
            shipvarn = conform(shipvar[:, fsegl:fsegr], stride=stride) # Get Along-track-averaged covariance profile (n-th segment).
            Lseg = Lsegs[n]
            # uQmeann = blkwavg(ushipn, coords, dim='x') # FIXME
            # Get along-track-averaged mean cross-track profile and covariance (for this segment).
            print(ushipn.shape, dxseg.shape, shipvarn.shape)
            uQmeann = np.sum(ushipn*dxseg, axis=1)*np.sum(shipvarn*dxseg, axis=1)/Lseg**2
            uQcovn = np.sum(ushipn*shipvarn*dxseg, axis=1)/Lseg
            uQeddyn = uQcovn - uQmeann # uQ = f(z).
            if kind=='eddytransp':     # uQ = f(segment #).
                dzsegavg = dzseg.mean(axis=1)
                uQmeann = np.sum(uQmeann*dzsegavg) # [u]*[Q]*m, transports per
                uQeddyn = np.sum(uQeddyn*dzsegavg) # unit along-track length.
            if kind=='eddyflux':
                if m==n==0:
                    uQmean = uQmeann[:,np.newaxis] # [u]*[Q], along-track
                    uQeddy = uQeddyn[:,np.newaxis] # averaged Q transports.
                else:
                    uQmean = np.hstack((uQmean, uQmeann[:,np.newaxis]))
                    uQeddy = np.hstack((uQeddy, uQeddyn[:,np.newaxis]))
            if kind=='eddytransp':
                if m==n==0:
                    uQmean = uQmeann
                    uQeddy = uQeddyn
                else:
                    uQmean = np.append(uQmean, uQmeann)
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
