# Description: Sample ROMS model output like a ship.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from ap_tools.utils import near, get_arrdepth
from stripack import trmesh
import pickle
from os.path import isfile
from pyroms.vgrid import z_r

__all__ = ['RomsShipSimulator']

class RomsShipSimulator(object):
    """
    USAGE
    -----
    shiproms = RomsShipSimulator(roms_fname, xship, yship, tship, verbose=True)

    Class that samples a ROMS *_his of *_avg output file simulating
    a ship track.
    """
    def __init__(self, roms_fname, xyship, tship):
        # Load track coordinates (x, y, t).
        assert xyship.size==tship.size, "(x,y,t) coordinates do not have the same size."
        xyship = self._burst(xyship)
        tship = self._burst(tship)
        self.xship = np.array([xy.lon for xy in xyship])
        self.yship = np.array([xy.lat for xy in xyship])
        self.tship = tship
        self.nshp = tship.size
        self._ndg = len(str(self.nshp))
        # Store roms grid (x, y, z, t) coordinates.
        self.filename = roms_fname
        self.nc = Dataset(self.filename)
        self.varsdict = self.nc.variables
        self.troms = self.varsdict['ocean_time']
        self.roms_time = self.troms[:] # Time in seconds from start of simulation.
        self.time_units = self.troms.units
        self.troms = num2date(self.troms[:], units=self.time_units)
        self.ship_time = date2num(self.tship, units=self.time_units)
        self.lonr = self.varsdict['lon_rho'][:]
        self.latr = self.varsdict['lat_rho'][:]
        self.lonu = self.varsdict['lon_u'][:]
        self.latu = self.varsdict['lat_u'][:]
        self.lonv = self.varsdict['lon_u'][:]
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


    def _interpxy(self, arrs, xn, yn, interpm, pointtype):
        """Interpolate model fields to ship (lon, lat) sample points."""
        arrs = np.array(arrs)

        if pointtype=='rho':
            mesh = self._trmesh_rho
        if pointtype=='u':
            mesh = self._trmesh_u
        if pointtype=='v':
            mesh = self._trmesh_v
        if pointtype=='psi':
            mesh = self._trmesh_psi

        if arrs.size==0:
            arrs = np.array([arrs]) # Array has to be at least 1D to iterate.

        intarr = []
        for arr in arrs:
            # print(xn.size,yn.size,arr.shape,arr.size,mesh.npts)
            intarr.append(mesh.interp(xn, yn, arr, order=interpm)[0])

        return np.array(intarr)


    def _interpt(self, arrs, ti):
        """Interpolate model fields to a given ship sample time."""
        arrs = np.array(arrs)

        # Make sure indices are increasing and not repeated.
        idl, idr = np.sort(near(self.troms, ti, npts=2, return_index=True))
        if idl==idr: idr+=1
        tli, tri = self.roms_time[idl], self.roms_time[idr]
        dti = (self.ship_time - tli)/(tri - tli)

        if arrs.size==0:
            arrs = np.array([arrs]) # Array has to be at least 1D to iterate.

        intarr = []
        for arr in arrs:
            arr_tl = arr[idl, :]
            arr_tr = arr[idr, :]
            z_tl = self.zr[idl, :]
            z_tr = self.zr[idr, :]
            intarr.append(arr_tl + (arr_tr - arr_tl)*dti) # Linearly interpolate in time.

        return np.array(intarr)


    def _interpsynop(self, arrs, ti, xn, yn, interpm, pointtype):
        """Interpolate model fields to a ship track pretending it was instantaneous."""
        intarr_synop = _interpxy(arrslr, xn, yn, interpm, pointtype)
        intarr_synop = _interpt(intarr_synop, arrs)

        return np.array(intarr)


    def plt_trkmap(self):
        """
        Plot topography map with the ship track.
        """
        fig, ax = plt.subplots()
        return fig, ax


    def plt_trkxyt(self):
        """
        Plot the ship track as a 3D line (x, y, t).
        """
        fig, ax = plt.subplots()
        return fig, ax


    def chk_synopticity(self, xn, yn, time, arr, varname, figsize=(16,8)):
        """
        Compare a ship-sampled transect of model output
        with the instantaneous (synoptic) model transect.

        # TODO: Plot vertical lines at the times where there is model output.
        """
        if arr.ndim==2:
            fig, ax = plt.subplots(ncols=3, figsize=figsize)
            ax1, ax2, ax3 = ax
            cs = ax1.contourf(tship3_num, zship3, var3dship, 30, shading='flat')
            ax1.axis('tight')
            ax1.set_ylabel('Depth [m]')
            plt.colorbar(cs)
            cs = ax1.contourf(tship3_num, zship3, var3dship, 30, shading='flat')
            ax1.axis('tight')
            ax1.set_ylabel('Depth [m]')
            plt.colorbar(cs)
            ax = (ax1, ax2, ax3)
        elif arr.ndim==1:
            fig, ax = plt.sublots(ncols=2, figsize=figsize)
            ax1, ax2 = ax
            fts = 1
            # Get the instantaneous line at the time the ship starts the line.

            ax1.plot(troms, romsarr, 'b-', marker='o')
            ax1.axis('tight')
            ax1.set_ylabel('Variable [var units]')
            ax = (ax1, ax2)
        plt.show()
        return fig, ax


    def ship_sample(self, varname, interp_method='linear', cache=False, verbose=True):
        """
        Interpolate model 'varname'
        to ship track coordinates (x, y, t).

        'interp_method' must be 'nearest',
        'linear' or 'cubic'.
        """
        deg2rad = np.pi/180 # [rad/deg].
        interpm = dict(nearest=0, linear=1, cubic=3)
        interpm = interpm[interp_method]
        xship_rad, yship_rad = self.xship*deg2rad, self.yship*deg2rad
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
                    # print(meshs, ptt)
                    cmd = "%s = trmesh(self.lon%s.ravel()*deg2rad, self.lat%s.ravel()*deg2rad)"%(trmeshs, ptt, ptt)
                    exec(cmd, loc_trmesh, locals())
                    cmd = "pickle.dump(%s, open(pklname, 'wb'))"%trmeshs
                    exec(cmd, loc_pickle, locals())
            else:
                cmd = "%s = trmesh(self.lon%s.ravel()*deg2rad, self.lat%s.ravel()*deg2rad)"%(trmeshs, ptt, ptt)
                exec(cmd, loc_trmesh, locals())

        # Store indices of adjacent time steps for each sample time.
        self.idxt = []
        idxtl, idxtr = [], []
        for t0 in self.tship.tolist():
            idl, idr = np.sort(near(self.troms, t0, npts=2, return_index=True)) # Make sure indices are increasing.
            if idl==idr: idr+=1 # Make sure indices are not repeated.
            self.idxt.append((idl, idr))
            idxtl.append(idl)
            idxtr.append(idr)

        vroms = self.varsdict[varname]
        if vroms.ndim==4:   # 4D (t, z, y, x) variables, like 'temp'.
            vship = np.empty((self.N, self.nshp))
            z_vship = np.empty((self.N, self.nshp))
        elif vroms.ndim==3: # 3D (x, y, t) variables, like 'zeta'.
            vship = np.empty((self.nshp))
        else:
            raise NotImplementedError('Not implemented yet, sorry...')

        tl, tr = self.roms_time[idxtl], self.roms_time[idxtr]
        self.dt = (self.ship_time - tl)/(tr - tl) # Store time separations.
        for n in range(self.nshp):
            if verbose:
                msg = (varname.upper(), str(n+1).zfill(self._ndg), str(self.nshp).zfill(self._ndg))
                print('Interpolating ship-sampled %s (%s of %s).'%msg)
            # Step 1: Find the time steps bounding the wanted time.
            var_tl = vroms[idxtl[n],:]
            var_tr = vroms[idxtr[n],:]
            z_tl = self.zr[idxtl[n],:]
            z_tr = self.zr[idxtr[n],:]
            xn, yn = xship_rad[n], yship_rad[n]
            tn, dtn = self.ship_time[n], self.dt[n]
            xn = np.array([xn, xn]) # Acoxambration (workaround) to avoid trmesh error.
            yn = np.array([yn, yn]) # Acoxambration (workaround) to avoid trmesh error.
            # Step 2: Horizontally interpolate the wanted variable
            # on each pair of bounding time steps and
            # Step 3: Interpolate the 2 time steps to the wanted
            # time in between them.
            if vroms.ndim==4:
                for nz in range(self.N):
                    vartup = (var_tl[nz,:].ravel(), var_tr[nz,:].ravel(), z_tl[nz,:].ravel(), z_tr[nz,:].ravel())
                    wrkl, wrkr, z_wrkl, z_wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                    vship[nz, n] = wrkl + (wrkr - wrkl)*dtn # Linearly interpolate in time.
                    z_vship[nz, n] = z_wrkl + (z_wrkr - z_wrkl)*dtn
            elif vroms.ndim==3:
                vartup = (var_tl.ravel(), var_tr.ravel())
                wrkl, wrkr = self._interpxy(vartup, xn, yn, interpm, pointtype)
                vship[n] = wrkl + (wrkr - wrkl)*dtn
            else:
                raise NotImplementedError('Not implemented yet, sorry...')

        # Convert to xarray.
        # coordd = {'lat':(['latitude','longitude'], y), 'lon':(['latitude','longitude'], x)}
        # dimsd = {'latitude':ny, 'longitude':nx}
        # Uii, Vii = Ui[0], Vi[0]
        # ui = xr.DataArray(Uii, coords=coordd, dims=dimsd)
        # vi = xr.DataArray(Vii, coords=coordd, dims=dimsd)
        # vard = dict(U=ui, V=vi)
        # UVii = xr.Dataset(vard, coords=coordd)

        # t_vship = np.tile(self.tship[np.newaxis,:], (self.N,1))

        # ds = xr.Dataset({'temperature': (['x', 'y', 'time'],  temp),
        #    ....:                  'precipitation': (['x', 'y', 'time'], precip)},
        #    ....:                 coords={'lon': (['x', 'y'], lon),
        #    ....:                         'lat': (['x', 'y'], lat),
        #    ....:                         'time': pd.date_range('2014-09-06', periods=3),
        #    ....:                         'reference_time': pd.Timestamp('2014-09-05')})

        if vroms.ndim==4:
            t_vship = np.tile(self.tship[np.newaxis,:], (self.N,1))
            return vship, t_vship, z_vship
        if vroms.ndim==3:
            t_vship = self.tship
            return vship, t_vship
