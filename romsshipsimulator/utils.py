# -*- coding: utf-8 -*-
#
# Description: Utilities for working with
#              the RomsShipSimulator class.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        May/2017

__all__ = ['mk_shiptrack']

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

    shipspd = shipspd*1852/3600 # Convert ship speed to m/s.
    sampfreq = sampfreq/3600 # convert sampling frequency to measurements/s.
    nwaypts = lons.size
    lontrk, lattrk = np.array([]), np.array([])
    dshp = shipspd/sampfreq # Spatial separation between measurements [m].
    for n in range(nwaypts-1):
        wptA = LatLon(lats[n], lons[n], height=0)
        wptB = LatLon(lats[n+1], lons[n+1], height=0)
        dAB = wptA.distanceTo(wptB) # length of current segment [m].
        dfrac = dAB/dshp # Separation between sample points as fraction of segment.
        nn = int(1/dfrac) - 1 # Number of points that fit in this segment (excluding waypoints A and B).
        if nn==-1:
            raise TrackError('Segment from %s to %s is too short to accomodate a ship data point.'%(wptA.toStr(), wptB.tostr()))
        ptsAB = [wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)]
        lontrk = np.append(lontrk, llon)
        lattrk = np.append(lattrk, llat)
    xship = np.pi
    yship = 1
    tship = 1
    return xship, yship, tship
