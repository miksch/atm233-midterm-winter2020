import xarray as xr
import datashader as ds
from datashader import transfer_functions as tf
import matplotlib.pyplot as plt

da = xr.open_dataset("data/dt_1_dz_0.001266667_data.nc")
da['temp'].plot.contourf()
plt.show()
#img = tf.shade(ds.Canvas(plot_height=400, plot_width=1200).raster(da['temp']))
#img