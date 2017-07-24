# Description: Sample ROMS model output like a ship.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from pandas import to_datetime
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from ap_tools.utils import xy2dist, near, fmt_isobath
from stripack import trmesh
import cartopy as ctpy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cmocean.cm import deep
import pickle
from os.path import isfile
from pyroms.vgrid import z_r
from pygeodesy.sphericalNvector import LatLon
from .utils import compass2trig, conform, isseq

__all__ = ['RomsShip',
           'ShipSample',
           'ShipTrack',
           'ShipTrackError']


class RomsShip(object):
    """
    USAGE
    -----
    shiproms = RomsShip(roms_fname, ship_track, verbose=True)

    Class that samples a ROMS *_his of *_avg output file simulating a
    ship track. The input 'ship_track' must be a 'letsroms.ShipTrack' instance.
    """
    def __init__(self, roms_fname, Shiptrack, verbose=True):
        assert isinstance(Shiptrack, ShipTrack), "Input must be a 'letsroms.ShipTrack' instance"
        iswaypt = [] # Flag the first and last points of a segment.
        tship = Shiptrack.trktimes.data
        xyship = Shiptrack.trkpts.data
        angship = Shiptrack.trkhdgs.data
        self.Shiptrack = Shiptrack # Attach the ShipTrack class.
        for ntrki in tship:
            iswpti = len(ntrki)*[0]
            iswpti[0], iswpti[-1] = 1, 1
            iswaypt.append(iswpti)
        self._deg2rad = np.pi/180  # [rad/deg].
        self._m2km = 1e-3          # [km/m].
        self._cph2hz = 1/3600      # [Hz/cph].
        self.iswaypt = np.bool8(self._burst(iswaypt))
        xyship = self._burst(xyship)
        self.tship = self._burst(tship)
        self.xship = np.array([xy.lon for xy in xyship])
        self.yship = np.array([xy.lat for xy in xyship])
        self._xship_rad = self.xship*self._deg2rad
        self._yship_rad = self.yship*self._deg2rad
        self.dship = xy2dist(self.xship, self.yship, datum='Sphere')
        self.angship = self._burst(angship)
        self.nshp = self.tship.size
        self._ndig = len(str(self.nshp))
        self.dx = self.dship[1:] - self.dship[:-1] # [m].
        self.dship = self.dship*self._m2km         # [km].
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
        self.lonmin = self.lonr.min()
        self.lonmax = self.lonr.max()
        self.latmin = self.latr.min()
        self.latmax = self.latr.max()
        self.bbox = [self.lonmin, self.lonmax, self.latmin, self.latmax]
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
        if verbose:
            print("Interpolating ROMS grid angle to ship track.")
        self.anggrdship = self.ship_sample('angle', interp_method='linear', \
                                           synop=True, segwise_synop=True, \
                                           cache=True, xarray_out=False, \
                                           verbose=False).vship/self._deg2rad

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
        if not isinstance(arrs[0], np.ndarray) \
        and not isinstance(arrs[0], list)\
        or isinstance(arrs[0], tuple):
            arrs = [arrs]

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
            intarr.append(mesh.interp(xn, yn, arr.ravel(), order=interpm)[0])

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

        intarr = []
        for arr in arrs:
            arr_tl = arr[idl, :]
            arr_tr = arr[idr, :]
            intarr.append(arr_tl + (arr_tr - arr_tl)*dti) # Linearly interpolate in time.

        return np.array(intarr)


    def plt_trkmap(self, isobaths=5, resolution='50m', borders=True, \
                   counties=False, rivers=True, topog='model', \
                   cmap=deep, ncf=100, trkcolor='r', trkmarker='o', trkms=5, \
                   trkmfc='r', trkmec='r', crs=ccrs.PlateCarree(), **kw):
        """
        Plot topography map with the ship track overlaid.
        """
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
        ax.set_extent(self.bbox, crs)
        LAND_hires = ctpy.feature.NaturalEarthFeature('physical', 'land', \
        resolution, edgecolor='k', facecolor=[.9]*3)
        ax.add_feature(LAND_hires, zorder=2, edgecolor='black')
        if borders:
            ax.add_feature(ctpy.feature.BORDERS, linewidth=0.5, zorder=3)
        if counties:
            provinces1 = ctpy.feature.NaturalEarthFeature('cultural', \
            'admin_1_states_provinces_lines', resolution, facecolor='none')
            provinces2 = ctpy.feature.NaturalEarthFeature('cultural', \
            'admin_2_states_provinces_lines', resolution, facecolor='none')
            ax.add_feature(provinces1, linewidth=0.5, zorder=3)
            ax.add_feature(provinces2, linewidth=0.5, zorder=3)
        if rivers:
            ax.add_feature(ctpy.feature.RIVERS, zorder=3)
        ax.coastlines(resolution, zorder=3)
        gl = ax.gridlines(draw_labels=True, zorder=5)
        gl.xlabels_top = False
        gl.ylabels_right = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER

        if topog: # Skip if don't want shaded topography.
            if topog=='model': # Plot model topography.
                ax.contourf(self.lonr, self.latr, self.h, ncf, cmap=cmap, zorder=0)
            elif isinstance(topog, tuple):      # Plot other topography, passed as a
                lontopo, lattopo, htopo = topog # (lon, lat, h) tuple.
                ax.contourf(lontopo, lattopo, htopo, ncf, cmap=cmap, zorder=0)

        if isobaths: # Skip if don't want any isobaths.
            if np.isscalar(isobaths): # Guess isobaths if not provided.
                hmi, hma = np.ceil(self.h.min()), np.floor(self.h.max())
                isobaths = np.linspace(hmi, hma, num=int(isobaths))
            elif isseq(isobaths):
                isobaths = list(isobaths)
            cc = ax.contour(self.lonr, self.latr, self.h, levels=isobaths, \
                            colors='grey', zorder=1)
            fmt_isobath(cc, manual=False)

        # Plot ship track.
        ax.plot(self.xship, self.yship, linestyle='-', color=trkcolor, \
                marker=trkmarker, ms=trkms, mfc=trkmfc, mec=trkmec, \
                zorder=4, **kw)

        return fig, ax


    def plt_trkxyt(self):
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


    def ship_sample(self, varname, interp_method='linear', synop=False, \
                    segwise_synop=False, cache=True, xarray_out=True, \
                    verbose=True, nprint=10):
        """
        Interpolate model 'varname' to ship track coordinates (x, y, t).
        Returns a 'ShipSample' object.

        'interp_method' must be 'nearest',
        'linear' or 'cubic'.

        If 'synop' is True (default False), the ship track is sampled
        instantaneously, depending on the value of 'segwise_synop'.

        If 'segwise_synop' is False (default), EACH OCCUPATION of the sample
        is synoptic, i.e., each set of consecutive lines that form the ship
        survey pattern. Otherwise, EACH TRANSECT is synoptic individually.

        TODO: Fix segwise_synop=True option.****
        """
        interpm = dict(nearest=0, linear=1, cubic=3)
        interpm = interpm[interp_method]
        self._interpm = interpm
        # Set up spherical Delaunay mesh to horizontally
        # interpolate the wanted variable in
        # the previous and next time steps.
        pointtype = self._get_pointtype(varname)
        ptt = pointtype[0]

        trmeshs = "self._trmesh_%s"%pointtype
        # Bring trmesh and pickle to the scope of the class.
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
        elif vroms.ndim==2: # 2D (x, y) variables, like 'angle'.
            vship = np.empty((self.nshp))
            vroms = vroms[:].ravel()
        else:
            print("Can only interpolate 2D, 3D or 4D variables.")
            return None

        if vroms.ndim>2:
            tl, tr = self.roms_time[idxtl], self.roms_time[idxtr]
            self.dt = np.abs(self.ship_time - tl)/(tr - tl) # Store time separations.
        msgn=nprint
        for n in range(self.nshp):
            if verbose and np.logical_or(msgn==nprint, n==self.nshp-1):
                msg = (varname.upper(), str(n+1).zfill(self._ndig), str(self.nshp).zfill(self._ndig))
                print('Ship-sampling %s (point %s of %s).'%msg)
                msgn=0
            msgn+=1
            # Step 1: Find the time steps bounding the wanted time.
            # (only for time-dependent variables).
            if vroms.ndim>2:
                var_tl = vroms[idxtl[n],:]
                var_tr = vroms[idxtr[n],:]
                tn, dtn = self.ship_time[n], self.dt[n]
            xn, yn = self._xship_rad[n], self._yship_rad[n]
            xn = np.array([xn, xn]) # Acoxambration (workaround) to avoid trmesh error.
            yn = np.array([yn, yn]) # Acoxambration (workaround) to avoid trmesh error.
            # Step 2: Horizontally interpolate the wanted variable
            # on each pair of bounding time steps and
            # Step 3: Interpolate the 2 time steps to the wanted
            # time in between them.
            if vroms.ndim==4: # 3D time-dependent variables (x, y, z, t).
                z_tl = self._get_vgridi(idxtl[n], pointtype)
                z_tr = self._get_vgridi(idxtr[n], pointtype)
                for nz in range(self.N):
                    vartup = (var_tl[nz,:].ravel(),
                              var_tr[nz,:].ravel(),
                              z_tl[nz,:].ravel(),
                              z_tr[nz,:].ravel())
                    wrkl, wrkr, z_wrkl, z_wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                    vship[nz, n] = wrkl + (wrkr - wrkl)*dtn # Linearly interpolate in time.
                    zvship[nz, n] = z_wrkl + (z_wrkr - z_wrkl)*dtn
                self.zship = zvship
            elif vroms.ndim==3: # 2D time-dependent variables (x, y, t).
                vartup = (var_tl.ravel(), var_tr.ravel())
                wrkl, wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                vship[n] = wrkl + (wrkr - wrkl)*dtn
            elif vroms.ndim==1: # 2D static variables (x, y). After ravel() a few lines above.
                vship[n] = self._interpxy((vroms), xn, yn, interpm, pointtype)

        # NOTE: Use pandas.to_datetime() to make time axis for xarray.
        # datetime.datetime() causes bugs in xarray.DataArray for
        # non-1D variables.
        if xarray_out:
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
                varunits = 'units not specified'
            attrsd = dict(units=varunits)
            # NOTE: DO NOT set 'encoding' kw on xr.DataArray. Causes
            # irreproducible (?) errors with the time coordinate.
            try:
                Vship = xr.DataArray(vship, coords=coordsd, dims=dimsd,
                                     name=varname.upper(), attrs=attrsd)
            except ValueError:
                print("Error converting variable '%s' to xarray.DataArray. Returning tuple of numpy.ndarray instead."%varname)
                if vship.ndim==2: Vship = (self.tship, self.zship, self.dship, \
                                           self.yship, self.xship, vship)
                elif vship.ndim==1: Vship = (self.tship, self.dship, \
                                             self.yship, self.xship, vship)
        else:
            if vship.ndim==2: Vship = (self.tship, self.zship, self.dship, \
                                       self.yship, self.xship, vship)
            elif vship.ndim==1: Vship = (self.tship, self.dship, self.yship, \
                                         self.xship, vship)

        return ShipSample(self, Vship)


class ShipSample(RomsShip):
    def __init__(self, Romsship, Vship):
        self.Romsship = Romsship # Attach parent class.
        self.dx = Romsship.dx
        self._interpm = self.Romsship._interpm
        Romsship.__delattr__('_interpm')
        if isinstance(Vship, xr.DataArray):
            self.ndim = Vship.ndim
            self.fromDataArray = True
            if self.ndim==2:
                self.zship = Vship.coords['depth']
                self.dz = self._strip(self.zship[1:,:] - self.zship[:-1,:])
                self.dz = 0.5*(self.dz[:,1:] + self.dz[:,:-1]) # [m].

            if Vship.attrs.__contains__('units') and len(Vship.attrs)>1:
                self.attrs_vship = Vship.attrs # Attach other attributes if they are present.
            self.name = Vship.name
            self.units = Vship.attrs['units']
            self.tship = Vship.coords['time']
            self.dship = Vship.coords['ship_dist']
            self.yship = Vship.coords['ship_lat']
            self.xship = Vship.coords['ship_lon']
            self.vship = Vship.data
            self.vship_DataArray = Vship # Keep full DataArray as well.

        elif isinstance(Vship, tuple):
            self.fromDataArray = False
            self.ndim = Vship[-1].ndim
            self.name = None
            self.units = None
            if self.ndim==1:
                self.tship, self.dship, \
                self.yship, self.xship, self.vship = Vship
            elif self.ndim==2:
                self.tship, self.zship, self.dship, \
                self.yship, self.xship, self.vship = Vship

        if self.ndim==1:
            self.dims = {'time':Romsship.nshp}
        elif self.ndim==2:
            self.dims = {'z':Romsship.N, 'time':Romsship.nshp}
            self.dz = self._strip(self.zship[1:,:] - self.zship[:-1,:])
            self.dz = conform(self.dz, stride='right') # [m].
            self.dx, self.dz = [np.array(arr) for arr in \
                                np.broadcast_arrays(self.dx[np.newaxis,:], \
                                                    self.dz)]
            self.dA = self.dx*self.dz # [m2].


    def add_noise(self, std, mean=0, kind='gaussian', verbose=True):
        """
        Add random noise of type 'kind' with amplitude 'std'
        and mean 'mean' (defaults to 0) to the ship-sampled variable.

        'kind' can be one of 'gaussian', 'white', 'red'
        or (?), (defaults to 'gaussian').
        """
        if kind=='gaussian':
            pdf = np.random.randn
        elif kind=='white':
            pdf = np.random.rand
        elif kind=='red':
            raise NotImplementedError("Red noise not implemented yet.")
            return

        # Add noise to the sampled variable.
        noise = std*pdf() + mean
        if hasattr(self, 'Vship'): # If data is a DataArray also add noise to it.
            self.Vship.data = self.Vship.data + noise
        self.vship = self.vship + noise
        if not hasattr(self, 'noise_properties'):
            self.noise_properties = dict(kind=[kind], amplitude=[std], mean=[mean])
        else:
            kind0 = noise_properties['kind']
            mean0 = noise_properties['mean']
            std0 = noise_properties['std']
            kind0.append(kind)
            mean0.append(mean)
            std0.append(std)
            self.noise_properties.update(dict(kind=kind0, amplitude=std0, mean=mean0))
        if verbose:
            msg = "Added %s noise with amplitude %.3f %s and mean %.3f %s to variable %s."%(kind, std, self.units, mean, self.units, self.name)
            print(msg.strip())

    def _strip(self, obj):
        if isinstance(obj, xr.DataArray):
            obj = obj.data
        else:
            pass

        return obj

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
                 will no longer reflect the ship speed exactly.
    closedtrk:   Whether to connect the last waypoint to the first.
    nrepeat:     Number of realizations of the track.
                 * If 'closedtrk' is 'False', moves the ship back and
                 forth along the waypoints, 'nrepeat' times.
                 * If 'closedtrk' is 'True', moves the ship along the
                 polygon formed by the waypoints, 'nrepeat' times.
    verbose:     Print track info to screen if 'True' (default).

    OUTPUT
    ------
    An instance of the 'letsroms.ShipTrack' class with the following attributes
    (and others):

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

        self.shipspd = xr.Variable(dims='', data=np.array([shipspd]),
                                   attrs=dict(units='kn, 1 kn = 1 nm/h = 1.852 km/h'))
        self.sampfreq = xr.Variable(dims='', data=np.array([sampfreq]),
                                    attrs=dict(units='measurements/h',
                                    long_name='Sampling frequency'))
        self.sampdt = xr.Variable(dims='', data=np.array([60/sampfreq]),
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
        self.dshp = xr.Variable(dims='', data=np.array([dshp]),
                                attrs=dict(units='m',
                                long_name='Separation between adjacent measurements'))
        trktimesi = tstart
        trktimes = []
        trkpts = []
        trkhdgs = []
        seg_lengths = []
        seg_times = []
        seg_npoints = []
        segment_index = np.array([])
        occupation_index = np.array([])
        for nrep in range(nrepeat):
            nrepp = nrep + 1
            if verbose:
                print("Realization %d/%d\n"%(nrepp, nrepeat))
            for n in range(self.nsegs):
                nsegp = n + 1
                wptA = LatLon(lats[n], lons[n])
                wptB = LatLon(lats[n+1], lons[n+1])
                dAB = wptA.distanceTo(wptB) # length of current segment [m].
                tAB = dAB/shipspd     # Occupation time of current segment [s].
                dfrac = dshp/dAB      # Separation between sample points
                                      # as a fraction of the segment.
                nn = int(1/dfrac) - 1 # Number of points that fit in this
                                      # segment (excluding waypoints A and B).
                if nn==-1:
                    raise ShipTrackError('Segment from %s to %s is not long \
                                         enough to accomodate any ship sampling \
                                         points.'%(wptA.toStr(), wptB.toStr()))
                if verbose:
                    print("Segment %d/%d:  %s --> %s (%.3f km | %.2f h)"\
                          %(n+1, self.nsegs, wptA.toStr(), wptB.toStr(), \
                          dAB*1e-3, tAB/3600))

                trkptsi = [wptA.intermediateTo(wptB, dfrac*ni) for ni in range(nn)]
                trktimesi = [trktimesi + timedelta(sampdt*ni/86400) for ni in range(nn)]

                # Fix actual start time of next segment by accounting for the
                # time to cover the distance between last sample point and wptB.
                ttransit = trkptsi[-1].distanceTo(wptB)/shipspd
                endsegtcorr = trktimesi[-1] + timedelta(ttransit/86400)
                trkptsi.append(wptB)
                trktimesi.append(endsegtcorr)
                nptsseg = nn + 1

                # Get headings in TRIG convention (East, North = 0, 90).
                As, Bs = trkptsi[:-1], trkptsi[1:]
                trkhdgsi = [compass2trig(a.bearingTo(b)) for a, b in zip(As, Bs)]
                trkhdgsi.append(compass2trig(As[-1].finalBearingTo(Bs[-1])))

                seg_npoints.append(nptsseg)
                trkhdgs.append(trkhdgsi) # Keep ship headings between each pair of points.
                trkpts.append(trkptsi)
                trktimes.append(trktimesi)
                trktimesi = endsegtcorr # Keep most recent time for next line.
                trkptsi = wptB          # Keep last point for next line.
                seg_lengths.append(dAB*1e-3)
                seg_times.append(tAB/3600)

                # Keep index of the current segment.
                segment_index = np.append(segment_index, \
                                          np.array([nsegp]*nptsseg))

                # Keep index of the current occupation as a coordinate.
                occupation_index = np.append(occupation_index, \
                                             np.array([nrepp]*nptsseg))
            if verbose:
                print("\n")

        # Store times and coordinates of the points along the track.
        attrspts = dict(long_name='Lon/lat coordinates of points sampled along the ship track')
        attrstimes = dict(long_name='Times of points sampled along the ship track')
        attrshdgs = dict(long_name='Angle from East to ship direction between points sampled along the ship track', units='degrees East')
        attrsocc = dict(long_name='Which occupation each sampled point belongs to')
        attrsseg = dict(long_name='Which segment each sampled point belongs to')
        assert len(trkpts)==len(trktimes)
        dim = 'point index'

        self.trkpts = xr.Variable(data=trkpts, dims=dim, attrs=attrspts)
        self.trktimes = xr.Variable(data=trktimes, dims=dim, attrs=attrstimes)
        self.trkhdgs = xr.Variable(data=trkhdgs, dims=dim, attrs=attrshdgs)
        self.occupation_index = xr.Variable(data=np.int32(occupation_index), \
                                            dims=dim, attrs=attrsocc)

        self.segment_index = xr.Variable(data=np.int32(segment_index), \
                                         dims=dim, attrs=attrsseg)

        segment_index = np.arange(self.nsegs*self.nrepeat) + 1
        seg_coords = {'segment index':segment_index}
        seg_dims = 'segment index'
        self.seg_lengths = xr.DataArray(seg_lengths, coords=seg_coords,
                                        dims=seg_dims, name='Length of each \
                                        segment of the track')
        self.seg_times = xr.DataArray(seg_times, coords=seg_coords, dims=seg_dims,
                                        name='Duration of each segment of the track')
        self.seg_npoints = xr.DataArray(seg_npoints, coords=seg_coords, dims=seg_dims,
                                        name='Number of points sampled on each \
                                        segment of the track')


class ShipTrackError(Exception):
    """
    Error raised when the ship track provided is invalid either because:

    1) The track does not close (if 'closedtrk' is 'True') or

    2) A segment of the ship track is too short to accomodate
    a single ship data point.
    """
    pass
