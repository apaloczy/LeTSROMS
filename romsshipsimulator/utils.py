# -*- coding: utf-8 -*-
#
# Description: Utilities for working with
#              the RomsShipSimulator class.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        May/2017

__all__ = ['mk_shiptrack', 'ShipTrackError']

import numpy as np
from pygeodesy.sphericalNvector import LatLon

def mk_shiptrack(waypts, sampfreq, shipspd=4, evenspacing=False, closedtrk=False, repeat=1):
    """
    Creates a ship track (longitude, latitude, time)
    to serve as input to a RomsShipSimulator instance.

    INPUT
    -----
    waypts:    A tuple of (longitude, latitude) arrays defining
               the waypoints of the track [degrees].
    sampfreq:  Ship sampling frequency (measurements/h).
    shipspd:   Underway speed [kn], 1 kn = 1 nm/h = 1.852 km/h ~ 0.5 m/s.
    closedtrk: Whether to connect the last waypoint to the first.
    repeat:    Number of occupations of the track.
               * If 'closedtrk' is 'False', moves the ship back and
               forth along the waypoints, 'repeat' times.
               * If 'closedtrk' is 'True', moves the ship along the
               polygon formed by the waypoints, 'repeat' times.

    OUTPUT
    ------
    xship:     Longitude of sample points along the track [degrees].
    yship:     Latitude of sample points along the track [degrees].
    tship:     Time of sample points along the track [secs from start].

    TODO
    ----
    Implement 'evenspacing' option.
    """
    lons, lats = map(np.array, (waypts[0], waypts[1]))
    assert lons.size==lats.size, "Different number of longitudes and latitudes."
    if closedtrk:
        assert np.logical_and(lons[0]==lons[-1], lats[0]==lats[-1]), "First and last waypoints must be identical for closedtrk==True."

    nwaypts = lons.size
    shipspd = shipspd*1852/3600 # Convert ship speed to m/s.
    sampfreq = sampfreq/3600 # convert sampling frequency to measurements/s.
    dshp = shipspd/sampfreq # Spatial separation between measurements [m].
    trkpts = []
    for n in range(nwaypts-1):
        wptA = LatLon(lats[n], lons[n])
        wptB = LatLon(lats[n+1], lons[n+1])
        dAB = wptA.distanceTo(wptB) # length of current segment [m].
        dfrac = dshp/dAB # Separation between sample points as fraction of segment.
        nn = int(1/dfrac) - 1 # Number of points that fit in this segment (excluding waypoints A and B).
        print(dAB, dfrac, dshp, nn)
        if nn==-1:
            raise ShipTrackError('Segment from %s to %s is too short to accomodate a ship data point.'%(wptA.toStr(), wptB.toStr()))
        trkpts.append([wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)])
    return trkpts
    # return xship, yship, tship

class ShipTrackError(Exception):
    """
    Error raised when a segment of the ship track is
    too short to accomodate a single ship data point.
    """
    pass
