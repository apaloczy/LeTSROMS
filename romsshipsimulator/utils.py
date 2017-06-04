# Description: Utilities for working with
#              the RomsShipSimulator class.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        May/2017

__all__ = ['mk_shiptrack', 'ShipTrackError']

import numpy as np
from pygeodesy.sphericalNvector import LatLon

def mk_shiptrack(waypts, ti, tf, sampfreq, shipspd=4, evenspacing=False, closedtrk=False, nrepeat=1, verbose=True):
    """
    Creates a ship track (longitude, latitude, time)
    to serve as input to a RomsShipSimulator instance.
aa
    INPUT
    -----
    waypts:      A tuple of (longitude, latitude) arrays defining
                 the waypoints of the track [degrees].
    tpts:        A tuple (start time, end time) of datetime objects defining
                 the start and end times of the track.
    sampfreq:    Ship sampling frequency (measurements/h).
    shipspd:     Underway speed [kn], 1 kn = 1 nm/h = 1.852 km/h ~ 0.5 m/s.
    closedtrk:   Whether to connect the last waypoint to the first.
    evenspacing: Whether to modify the ship sample points to be evenly spaced
                 within each line. The resulting track will no longer reflect
                 the ship speed.
    nrepeat:     Number of realizations of the track.
                 * If 'closedtrk' is 'False', moves the ship back and
                 forth along the waypoints, 'nrepeat' times.
                 * If 'closedtrk' is 'True', moves the ship along the
                 polygon formed by the waypoints, 'nrepeat' times.

    OUTPUT
    ------
    xship:       Longitude of sample points along the track [degrees].
    yship:       Latitude of sample points along the track [degrees].
    tship:       Time of sample points along the track [secs from start].

    TODO
    ----
    Implement 'evenspacing' option.
    """
    lons, lats = map(np.array, (waypts[0], waypts[1]))
    ti, tf = map(float, (ti, tf))

    if closedtrk and not np.logical_and(lons[0]==lons[-1], lats[0]==lats[-1]):
        raise ShipTrackError("First and last waypoints must be identical for closed ship tracks.")

    nwaypts = lons.size
    shipspd = shipspd*1852/3600 # Convert ship speed to m/s.
    sampfreq = sampfreq/3600 # convert sampling frequency to measurements/s.
    sampdt = 1/sampfreq # Time between measurements [s].
    dshp = shipspd/sampfreq # Spatial separation between measurements [m].
    trktimes = [ti]
    trkpts = []
    for nrep in range(nrepeat):
        if verbose:
            print("Realization %d/%d\n"%(nrep+1, nrepeat))
        nsegs = nwaypts - 1
        for n in range(nsegs):
            wptA = LatLon(lats[n], lons[n])
            wptB = LatLon(lats[n+1], lons[n+1])
            dAB = wptA.distanceTo(wptB) # length of current segment [m].
            tAB = dAB/shipspd # Occupation time of current segment [s].
            dfrac = dshp/dAB # Separation between sample points as a fraction of the segment.
            nn = int(1/dfrac) - 1 # Number of points that fit in this segment (excluding waypoints A and B).
            # print(round(dAB*1e-3, 1), round(dfrac, 3), nn).
            if nn==-1:
                raise ShipTrackError('Segment from %s to %s is not long enough to accomodate any ship sampling points.'%(wptA.toStr(), wptB.toStr()))
            if verbose:
                print("Segment %d/%d:  %s --> %s (%.3f km | %s h)"%(n+1, nsegs, wptA.toStr(), wptB.toStr(), dAB*1e-3, tAB/3600))
            trkpts.append([wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)])
            trktimes.append([trktimes[-1] + timedelta(sampdt*ni/86400) for ni in range(nn)])
        print("\n")

    trktimes.append(tf)
    return np.array(trkpts), np.array(trktimes)


class ShipTrackError(Exception):
    """
    Error raised when the ship track provided is invalid either because:

    1) The track does not close (if 'closedtrk' is 'True') or

    2) A segment of the ship track is too short to accomodate
    a single ship data point.
    """
    pass
