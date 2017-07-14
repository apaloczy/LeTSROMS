# Description: Utilities for working with
#              classes in LeTSROMS.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['crosstrk_flux',
           'pespec',
           'kespec',
           'compass2trig',
           'strip']

import numpy as np
from datetime import datetime, timedelta
from pygeodesy.utils import wrap180
from pygeodesy.sphericalNvector import LatLon
from xarray import DataArray, Variable
from ap_tools.utils import rot_vec


def crosstrk_flux(Romsship, varname, transp=False, synop=False, segwise_synop=False, **kw):
    """
    Calculate the cross-track mean (u_mean*Q_mean) eddy flux (u'*Q') for
    a variable "Q" sampled from ROMS output. All kwargs are
    passed to RomsShip.ship_sample().

    The mean is defined as the average of Q over each track line, such that
    u = u_mean + u' and Q = Q_mean + Q' are the total cross-track velocity
    and value of Q, respectively.

    if 'transp' is True (defaults to False), then the mean and eddy
    transports (i.e., the mean and eddy fluxes integrated in the vertical
    and in the along-track direction) for each segment of the track are
    returned instead of the fluxes. Returns flux(segment index) instead
    of flux(x, z).
    """
    assert isinstance(varname, str), "'varname' must be a string"
    # assert isinstance(Romsship, RomsShip), "Input must be a letsroms.RomsShip instance" # FIXME.

    uship = Romsship.ship_sample('u', synop=synop, segwise_synop=segwise_synop, **kw)
    vship = Romsship.ship_sample('v', synop=synop, segwise_synop=segwise_synop, **kw)
    shipvar = Romsship.ship_sample(varname, synop=synop, segwise_synop=segwise_synop, **kw)
    interpm = shipvar._interpm
    dA = shipvar.dA
    uship = strip(uship)
    vship = strip(vship)
    shipvar = strip(shipvar)

    # Rotate velocities to along-/across-track coordinates.
    # Rotate back to east-west coordinates and then to track coordinates.
    ang_trk = Romsship.angship
    ang_grd = Romsship.angle.ravel()/Romsship._deg2rad # [degrees].
    xsrad = strip(Romsship.xship)*Romsship._deg2rad
    ysrad = strip(Romsship.yship)*Romsship._deg2rad
    ang_grd =  Romsship._interpxy(ang_grd, xsrad, ysrad, interpm, 'rho')
    ang_tot = ang_trk - ang_grd # [degrees].
    _, uship = rot_vec(uship, vship, angle=ang_tot, degrees=True)
    uship = -uship # Cross-track velocity (u) is positive to the RIGHT of the track.

    # Calculate cross-track fluxes.
    for n in range(Romsship.Shiptrack.nsegs):
        n+=1
        fseg=Romsship.Shiptrack.segment_index==n
        print(uship.shape, shipvar.shape, fseg.shape, dA.shape)
        uQcov_a = uship[fseg]*shipvar[fseg]*dA[fseg]
        uQcov_avg = np.append(uQcov_avg, da*np.sum(uQcov_a)/da.sum())
    uship_bar, Qbar = uship*dA,

    return uQbar, uQeddy


def pespec():
    """Along-track PE spectrum"""
    return 1


def kespec():
    """Along-track PE spectrum"""
    return 1


def compass2trig(ang_compass, input_in_degrees=True):
    if not input_in_degrees:
        ang_compass = ang_compass*180/np.pi # Radians to degrees.
    aux = wrap180(ang_compass - 90)         # Move origin to Eastward direction.
    ang_trig = -aux                         # Increases couterclockwise (trig convention).

    return ang_trig


def strip(obj):
    if isinstance(obj, DataArray):
        obj = obj.vship_DataArray.data
    elif isinstance(obj, np.ndarray):
        pass
    else:
        obj = obj.vship

    return obj
