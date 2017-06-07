# LeTSROMS (Less Than Synoptic ROMS)

Sample ROMS model output like a ship/aircraft/drifter would. A helpful tool to plan field experiments.

# Dependencies

+ numpy
+ matplotlib
+ netCDF4
+ xarray
+ stripack
+ pyroms
+ pygeodesy

# Known issues

* Interpolation seems to break for u, v and v_bar, but not u_bar. Seems to be related to z_r.
