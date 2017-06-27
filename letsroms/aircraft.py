# Description: Sample ROMS model output like an aircraft.
# Author/date: André Palóczy, June/2017.
# E-mail:      paloczy@gmail.com

import numpy as np

__all__ = ['RomsAircraft']


class RomsAircraft(object):
    """
    USAGE
    -----
    planeroms = RomsAircraft(roms_fname, xaircraft, yaircraft, taircraft, verbose=True)

    Class that samples a ROMS *_his of *_avg output file simulating
    an aircraft track.
    """
    def __init__(self, roms_fname, xyaircraft, taircraft):
