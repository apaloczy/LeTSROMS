# Description: Sample ROMS model output like a ship.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        May/2017

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from ap_tools.utils import near, gen_dates
import pickle
from os.path import isfile
from stripack import trmesh
from pyroms.vgrid import z_r

__all__ = ['RomsShipSimulator']

class RomsShipSimulator(object):
    def __init__(self, roms_fname, xship, yship, tship, verbose=True):
        if verbose:
            print('Loading model grid and ship track.')
        # Store ship (x, y, z, t) coordinates.
        assert xship.size==yship.size==tship.size, "(x,y,t) coordinates do not have the same size."
        self.xship = xship
        self.yship = yship
        # self.zship = zship
        self.tship = tship
        self.nshp = tship.size
        # Store roms grid (x, y, z, t) coordinates.
        self.filename = roms_fname
        self.nc = Dataset(self.filename)
        self.varsdict = self.nc.variables
        timeroms = self.varsdict['ocean_time']
        self.time_units = timeroms.units
        self.troms = num2date(timeroms[:], units=self.time_units)
        self.roms_time = timeroms[:]
        self.ship_time = date2num(self.tship, units=self.time_units)
        self.lonr = self.varsdict['lon_rho'][:]
        self.latr = self.varsdict['lat_rho'][:]
        self.lonu = self.varsdict['lon_u'][:]
        self.latu = self.varsdict['lat_u'][:]
        self.lonv = self.varsdict['lon_u'][:]
        self.latv = self.varsdict['lat_v'][:]
        # self.lonp = self.varsdict['lon_psi'][:]
        # self.latp = self.varsdict['lat_psi'][:]
        self.angle = self.varsdict['angle'][:]
        self.h = self.varsdict['h'][:]
        self.hc = self.varsdict['hc'][:]
        self.s_rho = self.varsdict['s_rho'][:]
        self.Cs_r = self.varsdict['Cs_r'][:]
        self.N = self.Cs_r.size
        self.zeta = self.varsdict['zeta']
        self.Vtrans = self.varsdict['Vtransform'][:]
        self.zr = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)

    def plt_map(self):
        """
        Plot map with the ship track.
        """
        fig, ax = plt.subplots()
        return fig, ax

    def chk_synopticity(self):
        """
        Compare a ship-sampled transect of model output
        with the instantaneous (synoptic) model transect.
        """
        return 1

    def ship_sample(self, varname, interp_method='linear', cache=True, verbose=True):
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
        if not hasattr(self, '_trmesh_rho'):
            pklname = 'trmesh_cache-%s'%self.filename.split('/')[-1].replace('.nc','.pkl')
            if verbose:
                print('Setting up Delaunay mesh for RHO points.')
            if cache:
                if isfile(pklname): # Load from cache if it exists.
                    self._trmesh_rho = pickle.load(open(pklname, 'rb'))
                else: # Create cache if it does not exist.
                    self._trmesh_rho = trmesh(self.lonr.ravel()*deg2rad, self.latr.ravel()*deg2rad)
                    pickle.dump(self._trmesh_rho, open(pklname, 'wb'))
            else:
                self._trmesh_rho = trmesh(self.lonr.ravel()*deg2rad, self.latr.ravel()*deg2rad)

        # Locate bounding time steps for each ship time.
        self.idxt = []
        idxtl, idxtr = [], []
        for t0 in self.tship.tolist():
            idl, idr = np.sort(near(self.troms, t0, npts=2, return_index=True)) # Make sure indices are increasing.
            if idl==idr: # Make sure indices are not repeated.
                idr+=1
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
        dt = (self.ship_time - tl)/(tr - tl)
        for n in range(self.nshp):
            if verbose:
                print('Interpolating ship sample %d of %d.'%(n+1, self.nshp))
            var_tl = vroms[idxtl[n],:]
            var_tr = vroms[idxtr[n],:]
            z_tl = self.zr[idxtl[n],:]
            z_tr = self.zr[idxtr[n],:]
            xn, yn = xship_rad[n], yship_rad[n]
            tn, dtn = self.ship_time[n], dt[n]
            xn = np.array([xn, xn]) # Acoxambration to avoid trmesh error.
            yn = np.array([yn, yn]) # Acoxambration to avoid trmesh error.
            # Horizontally interpolate the wanted variable
            # on each pair of bounding time steps.
            if vroms.ndim==4:
                for nz in range(self.N):
                    wrkl = self._trmesh_rho.interp(xn, yn, var_tl[nz,:].ravel(), order=interpm)[0]
                    wrkr = self._trmesh_rho.interp(xn, yn, var_tr[nz,:].ravel(), order=interpm)[0]
                    z_wrkl = self._trmesh_rho.interp(xn, yn, z_tl[nz,:].ravel(), order=interpm)[0]
                    z_wrkr = self._trmesh_rho.interp(xn, yn, z_tr[nz,:].ravel(), order=interpm)[0]
                    vship[nz, n] = wrkl + (wrkr - wrkl)*dtn # Linearly interpolate in time.
                    z_vship[nz, n] = z_wrkl + (z_wrkr - z_wrkl)*dtn

            elif vroms.ndim==3:
                wrkl = self._trmesh_rho.interp(xn, yn, var_tl.ravel(), order=interpm)[0]
                wrkr = self._trmesh_rho.interp(xn, yn, var_tr.ravel(), order=interpm)[0]
                vship[n] = wrkl + (wrkr - wrkl)*dtn

        # Convert to xarray.
        # coordd = {'lat':(['latitude','longitude'], y), 'lon':(['latitude','longitude'], x)}
        # dimsd = {'latitude':ny, 'longitude':nx}
        # Uii, Vii = Ui[0], Vi[0]
        # ui = xr.DataArray(Uii, coords=coordd, dims=dimsd)
        # vi = xr.DataArray(Vii, coords=coordd, dims=dimsd)
        # vard = dict(U=ui, V=vi)
        # UVii = xr.Dataset(vard, coords=coordd)
        t_vship = np.tile(self.tship[np.newaxis,:], (self.N,1))

        # ds = xr.Dataset({'temperature': (['x', 'y', 'time'],  temp),
        #    ....:                  'precipitation': (['x', 'y', 'time'], precip)},
        #    ....:                 coords={'lon': (['x', 'y'], lon),
        #    ....:                         'lat': (['x', 'y'], lat),
        #    ....:                         'time': pd.date_range('2014-09-06', periods=3),
        #    ....:                         'reference_time': pd.Timestamp('2014-09-05')})

        if vroms.ndim==4:
            return vship, t_vship, z_vship
        if vroms.ndim==3:
            return vship, t_vship
