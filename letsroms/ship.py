# Description: Sample ROMS model output like a ship.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

import numpy as np
from matplotlib import pyplot as plt
from gsw import distance
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from pandas import to_datetime
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from ap_tools.utils import near, get_arrdepth
from stripack import trmesh
import pickle
from os.path import isfile
from pyroms.vgrid import z_r

__all__ = ['RomsShip']


class RomsShip(object):
    """
    USAGE
    -----
    shiproms = RomsShip(roms_fname, xship, yship, tship, verbose=True)

    Class that samples a ROMS *_his of *_avg output file simulating
    a ship track.
    """
    def __init__(self, roms_fname, xyship, tship):
        assert xyship.size==tship.size, "(x,y,t) coordinates do not have the same size."
        iswaypt = [] # Flag the first and last points of a segment.
        for ntrki in tship:
            iswpti = len(ntrki)*[0]
            iswpti[0], iswpti[-1] = 1, 1
            iswaypt.append(iswpti)
        self.iswaypt = np.bool8(self._burst(iswaypt))
        xyship = self._burst(xyship)
        self.tship = self._burst(tship)
        self.xship = np.array([xy.lon for xy in xyship])
        self.yship = np.array([xy.lat for xy in xyship])
        self.dship = np.append(0., np.cumsum(distance(self.xship, self.yship)))*1e-3
        self.nshp = self.tship.size
        self._ndig = len(str(self.nshp))
        self._deg2rad = np.pi/180  # [rad/deg].
        self.filename = roms_fname # Store roms grid (x, y, z, t).
        self.nc = Dataset(self.filename)
        self.varsdict = self.nc.variables
        self.troms = self.varsdict['ocean_time']
        self.roms_time = self.troms[:] # Time in seconds from start of simulation.
        self.time_units = self.troms.units
        self.calendar_type = self.troms.calendar
        self.troms = num2date(self.troms[:], units=self.time_units, calendar=self.calendar_type)
        self.ship_time = date2num(self.tship, units=self.time_units, calendar=self.calendar_type)
        self.lonr = self.varsdict['lon_rho'][:]
        self.latr = self.varsdict['lat_rho'][:]
        self.lonu = self.varsdict['lon_u'][:]
        self.latu = self.varsdict['lat_u'][:]
        self.lonv = self.varsdict['lon_v'][:]
        self.latv = self.varsdict['lat_v'][:]
        self.lonp = self.varsdict['lon_psi'][:]
        self.latp = self.varsdict['lat_psi'][:]
        self.angle = self.varsdict['angle'][:]
        self.h = self.varsdict['h'][:]
        self.hc = self.varsdict['hc'][:]
        self.s_rho = self.varsdict['s_rho'][:]
        self.Cs_r = self.varsdict['Cs_r'][:]
        self.N = self.Cs_r.size
        self.zeta = self.varsdict['zeta']
        self.Vtrans = self.varsdict['Vtransform'][:]
        self.zr = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        # Get vertical grid at interior u, v and psi points.
        hu = 0.5*(self.h[:,1:] + self.h[:,:-1])
        hv = 0.5*(self.h[1:,:] + self.h[:-1,:])
        hp = 0.5*(hv[:,1:] + hv[:,:-1])
        zetau = 0.5*(self.zeta[:,:,1:] + self.zeta[:,:,:-1])
        zetav = 0.5*(self.zeta[:,1:,:] + self.zeta[:,:-1,:])
        zetap = 0.5*(zetav[:,:,1:] + zetav[:,:,:-1])
        self.zu = z_r(hu, self.hc, self.N, self.s_rho, self.Cs_r, zetau, self.Vtrans)
        self.zv = z_r(hv, self.hc, self.N, self.s_rho, self.Cs_r, zetav, self.Vtrans)
        self.zp = z_r(hp, self.hc, self.N, self.s_rho, self.Cs_r, zetap, self.Vtrans)


    def _burst(self, arr):
        """
        Convert an array of lists of objects
        into a 1D array of objects.
        """
        arr = np.array(arr)
        wrk = np.array([])
        for n in range(arr.size):
            wrk = np.concatenate((wrk, np.array(arr[n])))

        return wrk


    def _get_pointtype(self, vname):
        """
        Given a ROMS variable name, returns
        the type of the grid point where it
        is located, either RHO, U, V or PSI.
        """
        vcoords = self.varsdict[vname].coordinates
        if 'lon_rho' in vcoords:
            ptype = 'rho'
        if 'lon_u' in vcoords:
            ptype = 'u'
        if 'lon_v' in vcoords:
            ptype = 'v'
        if 'lon_psi' in vcoords:
            ptype = 'psi'

        return ptype


    def _get_vgridi(self, n, pointtype):
        """
        Return the vertical grid at time step 'n'
        rho, u, v or psi points.
        """
        if pointtype=='rho':
            z = self.zr[n, :]
        if pointtype=='u':
            z = self.zu[n, :]
        if pointtype=='v':
            z = self.zv[n, :]
        if pointtype=='psi':
            z = self.zp[n, :]

        return z


    def _interpxy(self, arrs, xn, yn, interpm, pointtype):
        """Interpolate model fields to ship (lon, lat) sample points."""
        arrs = np.array(arrs)
        if arrs.size==0:
            arrs = np.array([arrs]) # Array has to be at least 1D to iterate.
        else:
            arrs = np.array(arrs)

        if pointtype=='rho':
            mesh = self._trmesh_rho
        if pointtype=='u':
            mesh = self._trmesh_u
        if pointtype=='v':
            mesh = self._trmesh_v
        if pointtype=='psi':
            mesh = self._trmesh_psi

        intarr = []
        for arr in arrs:
            intarr.append(mesh.interp(xn, yn, arr, order=interpm)[0])

        return np.array(intarr)


    def _interpt(self, arrs, ti, pointtype):
        """Interpolate model fields to a given ship sample time."""
        arrs = np.array(arrs)
        # Make sure indices are increasing and not repeated.
        idl, idr = np.sort(near(self.troms, ti, npts=2, return_index=True))
        if idl==idr: idr+=1

        tli, tri = self.roms_time[idl], self.roms_time[idr]
        dti = (self.ship_time - tli)/(tri - tli)

        if arrs.size==0:
            arrs = np.array([arrs]) # Array has to be at least 1D to iterate.
        else:
            arrs = np.array(arrs)

        z_tl = self._get_vgridi(idl, pointtype)
        z_tr = self._get_vgridi(idr, pointtype)

        intarr = []
        for arr in arrs:
            arr_tl = arr[idl, :]
            arr_tr = arr[idr, :]
            intarr.append(arr_tl + (arr_tr - arr_tl)*dti) # Linearly interpolate in time.

        return np.array(intarr)


    def plt_trkmap(self):
        """
        Plot topography map with the ship track.
        """
        fig, ax = plt.subplots()
        return fig, ax


    def plt_trkxyt(self, **kw):
        """
        Plot the ship track as a 3D line (x, y, t).

        All keyword arguments are passed to the 3D plot.
        """
        ship_days = (self.ship_time - self.ship_time[0])/86400.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.xship, self.yship, ship_days)
        ax.plot(self.xship[self.iswaypt], self.yship[self.iswaypt], ship_days[self.iswaypt], marker='o', ms=6)
        ax.set_xlabel('Ship longitude [degrees east]', fontsize=15, fontweight='black')
        ax.set_ylabel('Ship latitude [degrees north]', fontsize=15, fontweight='black')
        ax.set_zlabel('Ship time [cruise days]', fontsize=15, fontweight='black')

        return fig, ax


    def ship_sample(self, varname, interp_method='linear', synop=False, segwise_synop=False, cache=True, ret_xarray=True, verbose=True):
        """
        Interpolate model 'varname'
        to ship track coordinates (x, y, t).

        'interp_method' must be 'nearest',
        'linear' or 'cubic'.

        If 'synop' is True (default False), the ship track is sampled
        instantaneously, depending on the value of 'segwise_synop'.

        If 'segwise_synop' is False (default), EACH OCCUPATION of the sample
        is synoptic, i.e., each set of consecutive lines that form the ship
        survey pattern. Otherwise, EACH TRANSECT is synoptic individually.
        """
        interpm = dict(nearest=0, linear=1, cubic=3)
        interpm = interpm[interp_method]
        xship_rad, yship_rad = self.xship*self._deg2rad, self.yship*self._deg2rad
        # Set up spherical Delaunay mesh to horizontally
        # interpolate the wanted variable in
        # the previous and next time steps.
        pointtype = self._get_pointtype(varname)
        ptt = pointtype[0]

        stail = '.ravel(), order=interpm)[0]'
        trmeshs = "self._trmesh_%s"%pointtype
        # Bruing trmesh and pickle to the scope of the class.
        loc_trmesh = dict(trmesh=globals()['trmesh'])
        loc_pickle = dict(pickle=globals()['pickle'])
        if not hasattr(self, '_trmesh_%s'%pointtype):
            pklname = 'trmesh_%s-points_cache-%s'%(pointtype, self.filename.split('/')[-1].replace('.nc','.pkl'))
            if cache:
                if isfile(pklname): # Load from cache if it exists.
                    cmd = "%s = pickle.load(open(pklname, 'rb'))"%trmeshs
                    exec(cmd, loc_pickle, locals())
                else: # Create cache if it does not exist.
                    if verbose:
                        print('Setting up Delaunay mesh for %s points.'%pointtype.upper())
                    cmd = "%s = trmesh(self.lon%s.ravel()*self._deg2rad, self.lat%s.ravel()*self._deg2rad)"%(trmeshs, ptt, ptt)
                    exec(cmd, loc_trmesh, locals())
                    cmd = "pickle.dump(%s, open(pklname, 'wb'))"%trmeshs
                    exec(cmd, loc_pickle, locals())
            else:
                cmd = "%s = trmesh(self.lon%s.ravel()*self._deg2rad, self.lat%s.ravel()*self._deg2rad)"%(trmeshs, ptt, ptt)
                exec(cmd, loc_trmesh, locals())

        # Store indices of adjacent model time steps for the time of each sample.
        self.idxt = []
        idxtl, idxtr = [], []
        if synop:
            if segwise_synop: # Each TRANSECT of the track is instantaneous.
                waypts_idxs = np.where(self.iswaypt)[0].tolist()
            else: # Each OCCUPATION of the track is instantaneous.
                ndelim = np.logical_and(self.xship==self.xship[0],
                                        self.yship==self.yship[0])
                ndelim = np.where(ndelim)[0][:2].ptp() + 1
                fwaypts = np.where(self.iswaypt)[0]
                wptdl = np.arange(0, self.nshp+1, ndelim)
                wptdl[1:] = wptdl[1:] - 1
                wptdr = wptdl[1:-1] + 1
                waypts_idxs = np.concatenate((wptdl, wptdr)).tolist()
            waypts_idxs.sort()
            waypts_idxs.reverse()
            while len(waypts_idxs)>0:
                fsecl, fsecr = waypts_idxs.pop(), waypts_idxs.pop()
                self.tship[fsecl:fsecr+1] = self.tship[fsecl]
                self.ship_time[fsecl:fsecr+1] = self.ship_time[fsecl]
        else: # Non-synoptic sampling (i.e., ship-like, realistic).
            pass

        for t0 in self.tship.tolist():
            idl, idr = np.sort(near(self.troms, t0, npts=2, return_index=True)) # Make sure indices are increasing.
            if idl==idr: idr+=1 # Make sure indices are not repeated.
            self.idxt.append((idl, idr))
            idxtl.append(idl)
            idxtr.append(idr)

        vroms = self.varsdict[varname]
        if vroms.ndim==4:   # 4D (t, z, y, x) variables, like 'temp'.
            vship = np.empty((self.N, self.nshp))
            zvship = np.empty((self.N, self.nshp))
        elif vroms.ndim==3: # 3D (x, y, t) variables, like 'zeta'.
            vship = np.empty((self.nshp))
        else:
            print("Can only interpolate 3D or 4D variables.")
            return

        tl, tr = self.roms_time[idxtl], self.roms_time[idxtr]
        self.dt = np.abs(self.ship_time - tl)/(tr - tl) # Store time separations.
        for n in range(self.nshp):
            if verbose:
                msg = (varname.upper(), str(n+1).zfill(self._ndig), str(self.nshp).zfill(self._ndig))
                print('Ship-sampling %s (%s of %s).'%msg)
            # Step 1: Find the time steps bounding the wanted time.
            var_tl = vroms[idxtl[n],:]
            var_tr = vroms[idxtr[n],:]

            z_tl = self._get_vgridi(idxtl[n], pointtype)
            z_tr = self._get_vgridi(idxtr[n], pointtype)
            xn, yn = xship_rad[n], yship_rad[n]
            tn, dtn = self.ship_time[n], self.dt[n]
            xn = np.array([xn, xn]) # Acoxambration (workaround) to avoid trmesh error.
            yn = np.array([yn, yn]) # Acoxambration (workaround) to avoid trmesh error.
            # Step 2: Horizontally interpolate the wanted variable
            # on each pair of bounding time steps and
            # Step 3: Interpolate the 2 time steps to the wanted
            # time in between them.
            if vroms.ndim==4: # 3D variables.
                for nz in range(self.N):
                    vartup = (var_tl[nz,:].ravel(),
                              var_tr[nz,:].ravel(),
                              z_tl[nz,:].ravel(),
                              z_tr[nz,:].ravel())
                    wrkl, wrkr, z_wrkl, z_wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                    vship[nz, n] = wrkl + (wrkr - wrkl)*dtn # Linearly interpolate in time.
                    zvship[nz, n] = z_wrkl + (z_wrkr - z_wrkl)*dtn
                    self.zship = zvship
            elif vroms.ndim==3: # 2D variables.
                vartup = (var_tl.ravel(), var_tr.ravel())
                wrkl, wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                vship[n] = wrkl + (wrkr - wrkl)*dtn

        # NOTE: use pandas.to_datetime() to make time axis for xarray.
        # datetime.datetime() causes bugs in xarray.DataArray for
        # non-1D variables.
        if ret_xarray:
            tship_aux = to_datetime(self.tship)
            # Convert interpolated variables to xarray.
            if vship.ndim==2: # 4D variables (t,z,y,x) become time series of profiles (z,t).
                dimsd = {'z':self.N, 'time':self.nshp}
                dshp = np.tile(self.dship[np.newaxis,:], (self.N, 1))
                yshp = np.tile(self.yship[np.newaxis,:], (self.N, 1))
                xshp = np.tile(self.xship[np.newaxis,:], (self.N, 1))
                coordsd = {'time':tship_aux,
                           'depth':(['z', 'time'], self.zship),
                           'ship_dist':(['z', 'time'], dshp),
                           'ship_lat':(['z', 'time'], yshp),
                           'ship_lon':(['z', 'time'], xshp)}
            elif vship.ndim==1: # 3D variables (t,y,x) become time series (t).
                dimsd = {'time':self.nshp}
                coordsd = {'time':tship_aux,
                           'ship_dist':(['time'], self.dship),
                           'ship_lat':(['time'], self.yship),
                           'ship_lon':(['time'], self.xship)}

            # Make sure variable units, time units and calendar type are consistent.
            try:
                varunits = self.varsdict[varname].units
            except AttributeError:
                varunits = 'unitless'
            attrsd = dict(units=varunits)
            # NOTE: DO NOT set 'encoding' kw on xr.DataArray. Causes
            # irreproducible (?) errors with the time coordinate.
            try:
                Vship = xr.DataArray(vship, coords=coordsd, dims=dimsd,
                                     name=varname.upper(), attrs=attrsd)
            except ValueError:
                print("Error converting variable '%s' to xarray.DataArray. Returning tuple of numpy.ndarray instead."%varname)
                if vship.ndim==2: Vship = (self.tship, self.zship, self.dship, self.yship, self.xship, vship)
                elif vship.ndim==1: Vship = (self.tship, self.dship, self.yship, self.xship, vship)
        else:
            if vship.ndim==2: Vship = (self.tship, self.zship, self.dship, self.yship, self.xship, vship)
            elif vship.ndim==1: Vship = (self.tship, self.dship, self.yship, self.xship, vship)

        return Vship

        def add_noise(self, amp, std=0., type='white'):
            """
            Add random noise with amplitude 'amp' and standard deviation
            'std' (defaults to 0).
            """
            a
