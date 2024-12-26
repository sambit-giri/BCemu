import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
# import pkg_resources
from importlib.resources import files

def galaxy_clusters_Xray_data():
    '''
    X-ray data constraining gas properties inside galaxy clusters,
    taken from Sun et al (2009), Lovisari et al. (2015), and Gonzalez et al. (2013).
    '''
    # Path to the Excel file
    package_name = "BCemu"
    filename = str(files(package_name) / 'input_data/galaxy_clusters_gas_constraints.xlsx')

    # Load the Excel sheets into Pandas DataFrames
    Sun2009 = pd.read_excel(filename, sheet_name='Sun2009', skiprows=0)
    Lovisari2015 = pd.read_excel(filename, sheet_name='Lovisari2015', skiprows=0)
    Gonzalez2013 = pd.read_excel(filename, sheet_name='Gonzalez2013', skiprows=0)

    return {
        'Sun2009': Sun2009,
        'Lovisari2015': Lovisari2015,
        'Gonzalez2013': Gonzalez2013,
    }

def plot_galaxy_clusters_Xray_data(x='M500', y='fgas500', xerr='dM500', yerr='dfgas500', name='Sun2009',
                                   **kwargs):
    '''
    Plot X-ray data constraints on gas properties inside galaxy clusters,
    taken from Sun et al (2009), Lovisari et al. (2015) and Gonzalez et al. (2013).
    '''
    data_full = galaxy_clusters_Xray_data()
    assert name in data_full.keys(), f"name should be in {data_full.keys()}"
    data = data_full[name]
    xdata = np.array(data[x])[1:]
    ydata = np.array(data[y])[1:]
    xerrdata = np.array(data[xerr])[1:]
    yerrdata = np.array(data[yerr])[1:]
    xdata_unit = f'{x} [{np.array(data[x])[0]}]' if np.array(data[x])[0]==np.array(data[x])[0] else f'{x}'
    ydata_unit = f'{y} [{np.array(data[x])[0]}]' if np.array(data[y])[0]==np.array(data[y])[0] else f'{y}'

    xerrdata1, yerrdata1 = [], []
    for xd, yd in zip(np.array(data[xerr])[1:],np.array(data[yerr])[1:]):
        try:
            xerrdata1.append([xd.split(',')[0],xd.split(',')[1].split('−')[-1]])
        except:
            xerrdata1.append([xd,xd])
        try:
            yerrdata1.append([yd.split(',')[0],yd.split(',')[1].split('−')[-1]])
        except:
            yerrdata1.append([yd,yd])
    xerrdata = np.array(xerrdata1).astype(float)
    yerrdata = np.array(yerrdata1).astype(float)

    xdata, ydata, xerrdata, yerrdata = xdata[xdata!=-1], ydata[xdata!=-1], xerrdata[xdata!=-1], yerrdata[xdata!=-1]
    xdata, ydata, xerrdata, yerrdata = xdata[ydata!=-1], ydata[ydata!=-1], xerrdata[ydata!=-1], yerrdata[ydata!=-1]

    fig = kwargs.get('fig')
    ax  = kwargs.get('ax')
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1,figsize=(7,6))
    if ax is None: ax = fig.axes[0]
    ax.errorbar(xdata, ydata, xerr=np.abs(xerrdata).T, yerr=np.abs(yerrdata).T, ls=' ', label=name)
    ax.set_xlabel(xdata_unit)
    ax.set_ylabel(ydata_unit)
    ax.legend()
    if kwargs.get('show_plot') is not None: plt.show()
    return fig