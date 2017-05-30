# Description: Utilities for working with
#              the RomsShipSimulator class.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        May/2017

__all__ = ['mk_shiptrack', 'ShipTrackError']

import numpy as np
from pygeodesy.sphericalNvector import LatLon

def mk_shiptrack(waypts, sampfreq, shipspd=4, evenspacing=False, closedtrk=False, nrepeat=1, verbose=True):
    """
    Creates a ship track (longitude, latitude, time)
    to serve as input to a RomsShipSimulator instance.
aa
    INPUT
    -----
    waypts:    A tuple of (longitude, latitude) arrays defining
               the waypoints of the track [degrees].
    sampfreq:  Ship sampling frequency (measurements/h).
    shipspd:   Underway speed [kn], 1 kn = 1 nm/h = 1.852 km/h ~ 0.5 m/s.
    closedtrk: Whether to connect the last waypoint to the first.
    nrepeat:   Number of realizations of the track.
               * If 'closedtrk' is 'False', moves the ship back and
               forth along the waypoints, 'nrepeat' times.
               * If 'closedtrk' is 'True', moves the ship along the
               polygon formed by the waypoints, 'nrepeat' times.

    OUTPUT
    ------
    xship:     Longitude of sample points along the track [degrees].
    yship:     Latitude of sample points along the track [degrees].
    tship:     Time of sample points along the track [secs from start].

    TODO
    ----
    Implement 'evenspacing' option.
    Implement SHIP TIME CALCULATION and return it with lon, lats.
    """
    lons, lats = map(np.array, (waypts[0], waypts[1]))
    assert lons.size==lats.size, "Different number of longitudes and latitudes."

    if closedtrk and not np.logical_and(lons[0]==lons[-1], lats[0]==lats[-1]):
        raise ShipTrackError("First and last waypoints must be identical for closed ship tracks.")

    nwaypts = lons.size
    shipspd = shipspd*1852/3600 # Convert ship speed to m/s.
    sampfreq = sampfreq/3600 # convert sampling frequency to measurements/s.
    dshp = shipspd/sampfreq # Spatial separation between measurements [m].
    trkpts = []
    for nrep in range(nrepeat):
        if verbose:
            print("Realization %d/%d\n"%(nrep+1, nrepeat))
        nsegs = nwaypts - 1
        for n in range(nsegs):
            wptA = LatLon(lats[n], lons[n])
            wptB = LatLon(lats[n+1], lons[n+1])
            dAB = wptA.distanceTo(wptB) # length of current segment [m].
            dfrac = dshp/dAB # Separation between sample points as fraction of segment.
            nn = int(1/dfrac) - 1 # Number of points that fit in this segment (excluding waypoints A and B).
            # print(round(dAB*1e-3, 1), round(dfrac, 3), nn)
            if nn==-1:
                raise ShipTrackError('Segment from %s to %s is not long enough to accomodate any ship sampling points.'%(wptA.toStr(), wptB.toStr()))
            if verbose:
                print("Segment %d/%d:  %s --> %s (%.3f km),"%(n+1, nsegs, wptA.toStr(), wptB.toStr(), dAB*1e-3))
            trkpts.append([wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)])
        print("\n")

    return np.array(trkpts)
    # return xship, yship, tship

class ShipTrackError(Exception):
    """
    Error raised when a segment of the ship track is
    too short to accomodate a single ship data point.
    """
    pass
