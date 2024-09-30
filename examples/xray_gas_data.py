import numpy as np
import matplotlib.pyplot as plt
import BCemu 

xray_data_full = BCemu.galaxy_clusters_Xray_data()
fig = BCemu.plot_galaxy_clusters_Xray_data(x='M500', y='fgas500', xerr='dM500', yerr='dfgas500', 
                                           name='Sun2009', show_plot=True)