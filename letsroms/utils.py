# Description: Utilities for working with
#              LeTSROMS model-sampled fields.
# Author/date: André Palóczy, May/2017.
# E-mail:      paloczy@gmail.com

__all__ = ['blkwavg',
           'compass2trig',
           'strip',
           'conform',
           'isseq']

import numpy as np
from xarray import DataArray
from pygeodesy.utils import wrap180


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
    elif isinstance(obj, np.ndarray):
        pass
    else:
        obj = obj.vship

    return obj


def conform(arr, stride='right-up'):
    arr = np.array(arr)
    assert arr.ndim==2, "Array is not 2D."

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
