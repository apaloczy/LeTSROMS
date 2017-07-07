# Description: Utilities for working with
#              classes in LeTSROMS.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['ShipTrack',
           'ShipTrackError']

import numpy as np
from datetime import datetime, timedelta
from pygeodesy.sphericalNvector import LatLon
from xarray import DataArray, Variable


class ShipTrack(object):
    """
    USAGE
    -----
    ShipTrack(waypts, tstart, sampfreq=12, shipspd=4, evenspacing=False, closedtrk=False, nrepeat=1, verbose=True)

    Creates a ship track object (longitude, latitude, time)
    to serve as input to a RomsShip instance.

    INPUT
    -----
    waypts:      A tuple of (longitude, latitude) arrays defining
                 the waypoints of the track [degrees].
    tstart:      A datetime object defining the start time of the track.
    sampfreq:    Ship sampling frequency (measurements/h).
    shipspd:     Underway speed [kn], 1 kn = 1 nm/h = 1.852 km/h
                 ~ 0.5 m/s.
    evenspacing: Whether to modify the ship sample points to be
                 evenly spaced within each line. The resulting track
                 will no longer reflect the ship speed.
    closedtrk:   Whether to connect the last waypoint to the first.
    nrepeat:     Number of realizations of the track.
                 * If 'closedtrk' is 'False', moves the ship back and
                 forth along the waypoints, 'nrepeat' times.
                 * If 'closedtrk' is 'True', moves the ship along the
                 polygon formed by the waypoints, 'nrepeat' times.
    verbose:     Print track info to screen if 'True' (default).

    OUTPUT
    ------
    trkpts:      Array of lists of pygeodesy.sphericalNvector.LatLon
                 objects containing the (lon, lat) of each ship sample point along the track.
                 Each list is a segment of the track containing LatLon objects associated
                 with each ship sample point.
    trktimes:    Array of lists of datetime objects containing the
                 times of each ship sample point.

    TODO
    ----
    Implement 'evenspacing' option.
    """
    def __init__(self, waypts, tstart, sampfreq=12, shipspd=4, evenspacing=False, closedtrk=False, nrepeat=1, verbose=True):
        lons, lats = map(np.array, (waypts[0], waypts[1]))
        self.lons, self.lats = lons, lats

        if closedtrk and not np.logical_and(lons[0]==lons[-1], lats[0]==lats[-1]):
            raise ShipTrackError("First and last waypoints must be identical for closed ship tracks.")

        self.shipspd = Variable(dims='', data=np.array([shipspd]),
                                attrs=dict(units='kn, 1 kn = 1 nm/h = 1.852 km/h'))
        self.sampfreq = Variable(dims='', data=np.array([sampfreq]),
                                 attrs=dict(units='measurements/h',
                                 long_name='Sampling frequency'))
        self.sampdt = Variable(dims='', data=np.array([60/sampfreq]),
                               attrs=dict(units='minutes',
                               long_name='Time between adjacent measurements'))
        self.tstart = tstart
        self.is_evenspacing = evenspacing
        self.is_closedtrk = closedtrk
        self.nrepeat = nrepeat
        self.nwaypts = lons.size
        self.nsegs = self.nwaypts - 1
        shipspd = shipspd*1852/3600 # kn to m/s.
        sampfreq = sampfreq/3600    # measurements/h to measurements/s.
        sampdt = 1/sampfreq         # Time between adjacent measurements [s].
        dshp = shipspd/sampfreq     # Spatial separation between adjacent measurements [m].
        self.dshp = Variable(dims='', data=np.array([dshp]),
                             attrs=dict(units='m',
                             long_name='Separation between adjacent measurements'))
        trktimesi = tstart
        trktimes = []
        trkpts = []
        seg_lenghts = []
        seg_times = []
        seg_npoints = []
        occupation_number = np.array([])
        for nrep in range(nrepeat):
            nrepp = nrep + 1
            if verbose:
                print("Realization %d/%d\n"%(nrepp, nrepeat))
            for n in range(self.nsegs):
                wptA = LatLon(lats[n], lons[n])
                wptB = LatLon(lats[n+1], lons[n+1])
                dAB = wptA.distanceTo(wptB) # length of current segment [m].
                tAB = dAB/shipspd # Occupation time of current segment [s].
                dfrac = dshp/dAB  # Separation between sample points
                                  # as a fraction of the segment.
                nn = int(1/dfrac) - 1 # Number of points that fit in this
                                      # segment (excluding waypoints A and B).
                if nn==-1:
                    raise ShipTrackError('Segment from %s to %s is not long enough to accomodate any ship sampling points.'%(wptA.toStr(), wptB.toStr()))
                if verbose:
                    print("Segment %d/%d:  %s --> %s (%.3f km | %.2f h)"%(n+1, self.nsegs, wptA.toStr(), wptB.toStr(), dAB*1e-3, tAB/3600))
                trkptsi = [wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)]
                trktimesi = [trktimesi + timedelta(sampdt*ni/86400) for ni in range(nn)]
                # Fix actual start time of next segment by accounting for the
                # time to cover the distance between last sample point and wptB.
                ttransit = trkptsi[-1].distanceTo(wptB)/shipspd
                endsegtcorr = trktimesi[-1] + timedelta(ttransit/86400)
                trkptsi.append(wptB)
                trktimesi.append(endsegtcorr)
                nptsseg = nn + 1
                seg_npoints.append(nptsseg)
                trkpts.append(trkptsi)
                trktimes.append(trktimesi)
                trktimesi = endsegtcorr # Keep most recent time for next line.
                trkptsi = wptB          # Keep last point for next line.
                seg_lenghts.append(dAB*1e-3)
                seg_times.append(tAB/3600)
                # Keep number of the current occupation as a coordinate.
                occupation_number = np.append(occupation_number,
                                              np.array([nrepp]*nptsseg))
            if verbose:
                print("\n")

        # Store times and coordinates of the points along the track.
        attrspts = dict(long_name='Lon/lat coordinates of points sampled along the ship track')
        attrstimes = dict(long_name='Times of points sampled along the ship track')
        attrsocc = dict(long_name='')
        assert len(trkpts)==len(trktimes)
        dim = 'point number'

        self.trkpts = Variable(data=trkpts, dims=dim, attrs=attrspts)
        self.trktimes = Variable(data=trktimes, dims=dim, attrs=attrstimes)
        self.occupation_number = Variable(data=occupation_number, dims=dim,
                                          attrs=attrsocc)

        segment_number = np.arange(self.nsegs*self.nrepeat) + 1
        seg_coords = {'segment number':segment_number}
        seg_dims = 'segment number'
        self.seg_lenghts = DataArray(seg_lenghts, coords=seg_coords,
                                     dims=seg_dims, name='Length of each track segment')
        self.seg_times = DataArray(seg_times, coords=seg_coords, dims=seg_dims,
                                   name='Duration of each track segment')
        self.seg_npoints = DataArray(seg_npoints, coords=seg_coords,
                                     dims=seg_dims,
                                     name='Number of points sampled on each track segment')

    def xtrk_flux(self, varname, transp=False, synop=False, segwise_synop=False, **kw):
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
        return 1


class ShipTrackError(Exception):
    """
    Error raised when the ship track provided is invalid either because:

    1) The track does not close (if 'closedtrk' is 'True') or

    2) A segment of the ship track is too short to accomodate
    a single ship data point.
    """
    pass
