# Description: Utilities for working with
#              LeTSROMS model-sampled fields.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['mk_transect',
           'mk_basemap',
           'blkwavg',
           'compass2trig',
           'strip',
           'conform',
           'isseq']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from xarray import DataArray, Variable
from cmocean.cm import deep
from ap_tools.utils import xy2dist, fmt_isobath
import cartopy as ctpy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pygeodesy.utils import wrap180

km2nm = 1/1.852 # [nm/km]
m2km = 1e-3     # [km/m]


def mk_transect(ax, ntransects, contiguous=True, shipspd=None, \
                bbox_zoom=None, crs=ccrs.PlateCarree()):
    """
    USAGE
    -----
    lon, lat, dist = mk_transect(ax, ntransects, plot=True)
    """
    if bbox_zoom:
        ax.set_extent(bbox_zoom, crs)
        plt.draw()
    ninp = 2
    Xs, Ys, Ds, Ts = [], [], [], []
    for n in range(ntransects):
        if ninp==2:
            xs, ys = np.array([]), np.array([])
        pts = plt.ginput(n=ninp, timeout=0)
        for pt in pts:
            x0, y0 = pt[0], pt[1]
            xs = np.append(xs, x0)
            ys = np.append(ys, y0)
        if contiguous:
            ninp = 1

        dd =  xy2dist(xs, ys, datum='Sphere')[-1]*m2km # [km].
        print("Transect length: %.2f km / %.2f nm"%(dd, dd*km2nm))
        if shipspd:
            dt = dd*km2nm/shipspd # [h]
            print("Occupation time at %.1f kn: %.2f h"%(shipspd, dt))
            Ts.append(dt)
        Xs.append(xs)
        Ys.append(ys)
        Ds.append(dd)

        ax.plot(xs, ys, 'r', linewidth=2.0, zorder=9)
        plt.draw()

    Xs, Ys, Ds, Ts = map(np.squeeze, (Xs, Ys, Ds, Ts))
    Xs, Ys, Ds, Ts = map(np.array, (Xs, Ys, Ds, Ts))
    if contiguous:
        Xs, Ys = Xs[-1], Ys[-1]

    if shipspd:
        print('')
        print("Total occupation time at %.1f kn ---------> %.2f h"%(shipspd, Ts.sum()))
        return Xs, Ys, Ds, Ts
    else:
        return Xs, Ys, Ds


def mk_basemap(bbox, bbox_zoom=None, topog=None, topog_style='contour', \
               which_isobs=3, resolution='50m', borders=True, counties=False, \
               rivers=True, xycoast=None, cmap=deep, ncf=100, \
               manual_clabel=False, crs=ccrs.PlateCarree()):
    """
    USAGE
    -----
    fig, ax = mk_basemap(bbox, **kw)

    Makes a base map covering the given 'bbox' [lonmin, lonmax, latmin, latmax].
    """
    if bbox_zoom:
        bbox = bbox_zoom

    fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
    ax.set_extent(bbox, crs)
    if xycoast:
        if isinstance(xycoast, Polygon):
            coast = xycoast
        else:
            xcoast, ycoast = xycoast
            coast = Polygon(np.hstack((xcoast[:,np.newaxis], ycoast[:,np.newaxis])), closed=True, fill=True, facecolor='lightgray', edgecolor='k')
        ax.add_patch(coast)
    else:
        LAND_hires = ctpy.feature.NaturalEarthFeature('physical', 'land', \
        resolution, edgecolor='k', facecolor=[.9]*3)
        ax.add_feature(LAND_hires, zorder=2, edgecolor='black')
        ax.coastlines(resolution, zorder=3)

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
    if isinstance(topog, tuple):        # Plot topography passed as a
        lontopo, lattopo, htopo = topog # (lon, lat, h) tuple.
    if topog is None:
        pass
    else:
        raise IOError("'topog' must be a (lon, lat, topo) tuple or None.")

    if topog is not None:
        if topog_style=='contour' or topog_style=='both':
            if np.isscalar(which_isobs): # Guess isobaths if not provided.
                hmi, hma = np.ceil(htopo.min()), np.floor(htopo.max())
                which_isobs = np.linspace(hmi, hma, num=int(which_isobs)).tolist()
            elif isseq(which_isobs):
                which_isobs = list(which_isobs)
            cc = ax.contour(lontopo, lattopo, htopo, levels=which_isobs, \
                            colors='grey', zorder=1)
            fmt_isobath(cc, manual=manual_clabel, zorder=0)
        if topog_style=='pcolor' or topog_style=='both':
            ax.pcolormesh(lontopo, lattopo, htopo, cmap=cmap, zorder=0)
        if topog_style=='contourf':
            ax.contourf(lontopo, lattopo, htopo, ncf, cmap=cmap, zorder=0)

    gl = ax.gridlines(draw_labels=True, zorder=5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gltl = ax.xaxis.get_majorticklabels()
    ax.xaxis.set_ticklabels(gltl, rotation=45)
    plt.draw()

    return fig, ax


def blkwavg(arr, coords, dim='x', bins='coarsest', ret_integral=False, **kw):
    if bins=='coarsest':
        coords = 1
        bins = linspace()
    else:
        raise NotImplementedError
    arr = DataArray(data=arr, coords=coords)
    arr.groupby_bins(dim, bins, right=True, labels=newax, **kw).mean(dim)

    if not ret_integral:
        arr = arr/bins.sum()

    return None


def compass2trig(ang_compass, input_in_degrees=True):
    if not input_in_degrees:
        ang_compass = ang_compass*180/np.pi # Radians to degrees.
    aux = wrap180(ang_compass - 90)         # Move origin to Eastward direction.
    ang_trig = -aux                         # Increases couterclockwise (trig convention).

    return ang_trig


def strip(obj):
    if isinstance(obj, DataArray):
        obj = obj.vship_DataArray.data
    elif isinstance(obj, Variable):
        obj = obj.data
    elif isinstance(obj, np.ndarray):
        pass
    else:
        obj = obj.vship

    return obj


def conform(arr, stride='right-up'):
    if not stride:
        return arr
    arr = np.array(arr)
    if stride=='right-up' or stride=='up':
        assert arr.ndim==2, "Array is not 2D."

    if arr.ndim==1:
        if stride=='right' or stride=='right-up':
            arr = 0.5*(arr[1:] + arr[:-1])
        if stride=='up' or stride=='right-up':
            arr = 0.5*(arr[1:] + arr[:-1])
    elif arr.ndim==2:
        if stride=='right' or stride=='right-up':
            arr = 0.5*(arr[:,1:] + arr[:,:-1])
        if stride=='up' or stride=='right-up':
            arr = 0.5*(arr[1:,:] + arr[:-1,:])

    return arr


def isseq(obj):
    isseq = isinstance(obj, list) or \
            isinstance(obj, tuple) or \
            isinstance(obj, np.ndarray)

    return isseq
