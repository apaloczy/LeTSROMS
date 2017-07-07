# Description: Utilities for working with
#              classes in LeTSROMS.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['crosstrk_flux',
           'compass2trig']

import numpy as np
from datetime import datetime, timedelta
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
    and in the along-track direction) are returned instead of the fluxes.
    """
    # Get synoptic sample.
    usynop = Romsship.ship_sample('u', synop=True, \
                                  segwise_synop=segwise_synop, **kw)
    vsynop = Romsship.ship_sample('v', synop=True, \
                                  segwise_synop=segwise_synop, **kw)
    synopflx = Romsship.ship_sample(varname, synop=True, \
                                    segwise_synop=segwise_synop, **kw)
    # Get quasi-synoptic sample.
    uship = Romsship.ship_sample('u', synop=False, **kw)
    vship = Romsship.ship_sample('v', synop=False, **kw)
    shipflx = Romsship.ship_sample(varname, synop=False, **kw)

    # Rotate velocities to along-/across-track coordinates.
    ang_trk = Romsship.Shiptrack.
    usynop, vsynop = rot_vec(usynop, vsynop, angle=ang_trk, degrees=False)
    uship, vship = rot_vec(uship, vship, angle=ang_trk, degrees=False)

    return 1


    def compass2trig(ang_compass):
        aux = ang_compass - 90         # Move origin to Eastward direction.
        aux = wrap180(aux)
        ang_trig = -aux # Increases couterclockwise (trig convention).

        return ang_trig
