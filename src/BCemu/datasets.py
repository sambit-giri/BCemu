import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
# import pkg_resources
from importlib.resources import files
from .download import get_package_resource_path, download_hydrosims

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


def _hydrosims_dir():
    base = get_package_resource_path('BCemu', 'input_data')
    return os.path.join(base, 'HydroSims')

def _ensure_hydrosims():
    d = _hydrosims_dir()
    if not os.path.exists(d) or len(os.listdir(d)) < 3:
        download_hydrosims()


class HydroSimDataPk:
    """
    Load raw matter power spectrum P(k, z) from hydrodynamical simulations.

    Available simulations include BAHAMAS, OWLS, C-OWLS, DMONLY power tables
    and FLAMINGO HDF5 runs (both hydrodynamical and DMO variants).

    Parameters
    ----------
    name : str
        Simulation name. Use ``HydroSimDataPk.available()`` for the full list.

    Examples
    --------
    >>> sim = HydroSimDataPk('FLAMINGO_L1_m9')
    >>> data = sim.load()   # {'z': ..., 'k': ..., 'Pk': ...}
    """

    _FLAMINGO_HDF5 = {
        'FLAMINGO_L1_m9':         'L1_m9.hdf5',
        'FLAMINGO_L1_m9_DMO':     'L1_m9_DMO.hdf5',
        'FLAMINGO_L1_m8':         'L1_m8.hdf5',
        'FLAMINGO_L1_m8_DMO':     'L1_m8_DMO.hdf5',
        'FLAMINGO_Jet':           'Jet.hdf5',
        'FLAMINGO_Jet_fgas-4sigma': 'Jet_fgas-4σ.hdf5',
        'FLAMINGO_Planck_DMO':    'Planck_DMO.hdf5',
        'FLAMINGO_fgas+2sigma':   'fgas+2σ.hdf5',
        'FLAMINGO_fgas-2sigma':   'fgas-2σ.hdf5',
        'FLAMINGO_fgas-4sigma':   'fgas-4σ.hdf5',
        'FLAMINGO_fgas-8sigma':   'fgas-8σ.hdf5',
    }

    def __init__(self, name):
        self.name = name
        self._data = None

    @staticmethod
    def available():
        """Return sorted list of available simulation names."""
        _ensure_hydrosims()
        d = _hydrosims_dir()
        powtable = sorted(
            f[len('powtable_'):-len('.dat')]
            for f in os.listdir(d)
            if f.startswith('powtable_') and f.endswith('.dat')
        )
        return powtable + sorted(HydroSimDataPk._FLAMINGO_HDF5.keys())

    def load(self):
        """
        Load P(k, z) data.

        Returns
        -------
        dict
            Keys: ``z`` (1-D array), ``k`` (1-D array, h/Mpc),
            ``Pk`` (2-D array, shape ``[n_z, n_k]``, (Mpc/h)^3).
        """
        if self._data is not None:
            return self._data
        _ensure_hydrosims()
        d = _hydrosims_dir()
        if self.name in self._FLAMINGO_HDF5:
            path = os.path.join(d, 'FLAMINGO', self._FLAMINGO_HDF5[self.name])
            self._data = self._load_flamingo_hdf5(path)
        else:
            path = os.path.join(d, f'powtable_{self.name}.dat')
            if not os.path.exists(path):
                avail = self.available()
                raise ValueError(f"'{self.name}' not found. Available: {avail}")
            self._data = self._load_powtable(path)
        return self._data

    @staticmethod
    def _load_powtable(path):
        data = np.genfromtxt(path, comments='#')
        z_all, k_all, Pk_all = data[:, 0], data[:, 1], data[:, 2]
        z_vals = np.unique(z_all)
        k_vals = np.unique(k_all)
        Pk = np.zeros((len(z_vals), len(k_vals)))
        for i, z in enumerate(z_vals):
            mask = z_all == z
            idx = np.argsort(k_all[mask])
            Pk[i] = Pk_all[mask][idx]
        return {'z': z_vals, 'k': k_vals, 'Pk': Pk}

    @staticmethod
    def _load_flamingo_hdf5(path):
        import h5py
        with h5py.File(path, 'r') as f:
            z_keys = sorted([k for k in f.keys() if k.startswith('z=')])
            z_vals = np.array([float(k[2:]) for k in z_keys])
            sort_idx = np.argsort(z_vals)
            z_vals = z_vals[sort_idx]
            z_keys = [z_keys[i] for i in sort_idx]
            k_vals = np.array(f[z_keys[0]]['k'])
            Pk = np.zeros((len(z_vals), len(k_vals)))
            for i, key in enumerate(z_keys):
                Pk[i] = np.array(f[key]['P(k)'])
        return {'z': z_vals, 'k': k_vals, 'Pk': Pk}


class HydroSimDataSk:
    """
    Load matter power spectrum suppression S(k, z) = P_bary / P_dm from
    hydrodynamical simulations.

    Available simulations span EAGLE, Illustris, TNG100, TNG300, MB2,
    HorizonAGN, BAHAMAS (AGN variants), and FLAMINGO (all hdf5 models).

    Parameters
    ----------
    name : str
        Simulation name. Use ``HydroSimDataSk.available()`` for the full list.

    Examples
    --------
    >>> sim = HydroSimDataSk('FLAMINGO_L1_m9')
    >>> data = sim.load()   # {'z': ..., 'k': ..., 'Sk': ...}
    """

    _NPZ_SIMS = {
        'HorizonAGN': 'HorizonAGN_data.npz',
        'TNG300':     'TNG300_data.npz',
    }

    _LOGPK_SIMS = {
        'TNG100':          'logPkRatio_TNG100.dat',
        'EAGLE':           'logPkRatio_eagle.dat',
        'Illustris':       'logPkRatio_illustris.dat',
        'MB2':             'logPkRatio_mb2.dat',
        'HorizonAGN_HzAGN': 'logPkRatio_HzAGN.dat',
    }

    _BAHAMAS_PAIRS = {
        'BAHAMAS_AGN7.6': ('BAHAMAS_AGN7.6_matter', 'BAHAMAS_AGN7.6_DMONLY'),
        'BAHAMAS_AGN7.8': ('BAHAMAS_AGN7.8_matter', 'BAHAMAS_AGN7.8_DMONLY'),
        'BAHAMAS_AGN8.0': ('BAHAMAS_AGN8.0_matter', 'BAHAMAS_AGN8.0_DMONLY'),
    }

    # (hydro hdf5 filename, DMO hdf5 filename)
    _FLAMINGO_PAIRS = {
        'FLAMINGO_L1_m9':           ('L1_m9.hdf5',          'L1_m9_DMO.hdf5'),
        'FLAMINGO_L1_m8':           ('L1_m8.hdf5',          'L1_m8_DMO.hdf5'),
        'FLAMINGO_Jet':             ('Jet.hdf5',             'L1_m9_DMO.hdf5'),
        'FLAMINGO_Jet_fgas-4sigma': ('Jet_fgas-4σ.hdf5', 'L1_m9_DMO.hdf5'),
        'FLAMINGO_fgas+2sigma':     ('fgas+2σ.hdf5',   'L1_m9_DMO.hdf5'),
        'FLAMINGO_fgas-2sigma':     ('fgas-2σ.hdf5',   'L1_m9_DMO.hdf5'),
        'FLAMINGO_fgas-4sigma':     ('fgas-4σ.hdf5',   'L1_m9_DMO.hdf5'),
        'FLAMINGO_fgas-8sigma':     ('fgas-8σ.hdf5',   'L1_m9_DMO.hdf5'),
    }

    def __init__(self, name):
        self.name = name
        self._data = None

    @staticmethod
    def available():
        """Return sorted list of available simulation names."""
        _ensure_hydrosims()
        return (
            sorted(HydroSimDataSk._NPZ_SIMS.keys())
            + sorted(HydroSimDataSk._LOGPK_SIMS.keys())
            + sorted(HydroSimDataSk._BAHAMAS_PAIRS.keys())
            + sorted(HydroSimDataSk._FLAMINGO_PAIRS.keys())
        )

    def load(self):
        """
        Load matter power suppression S(k, z) = P_bary / P_dm.

        Returns
        -------
        dict
            Keys: ``z`` (1-D array), ``k`` (1-D array, h/Mpc or Mpc^-1
            depending on source), ``Sk`` (2-D array, shape ``[n_z, n_k]``).
        """
        if self._data is not None:
            return self._data
        _ensure_hydrosims()
        d = _hydrosims_dir()

        if self.name in self._NPZ_SIMS:
            path = os.path.join(d, self._NPZ_SIMS[self.name])
            raw = np.load(path)
            self._data = {'z': raw['z'], 'k': raw['k'], 'Sk': raw['Sk']}

        elif self.name in self._LOGPK_SIMS:
            path = os.path.join(d, self._LOGPK_SIMS[self.name])
            self._data = self._load_logpk(path)

        elif self.name in self._BAHAMAS_PAIRS:
            matter_name, dmo_name = self._BAHAMAS_PAIRS[self.name]
            m = HydroSimDataPk._load_powtable(os.path.join(d, f'powtable_{matter_name}.dat'))
            dm = HydroSimDataPk._load_powtable(os.path.join(d, f'powtable_{dmo_name}.dat'))
            self._data = {'z': m['z'], 'k': m['k'], 'Sk': m['Pk'] / dm['Pk']}

        elif self.name in self._FLAMINGO_PAIRS:
            hydro_fn, dmo_fn = self._FLAMINGO_PAIRS[self.name]
            hydro = HydroSimDataPk._load_flamingo_hdf5(os.path.join(d, 'FLAMINGO', hydro_fn))
            dmo   = HydroSimDataPk._load_flamingo_hdf5(os.path.join(d, 'FLAMINGO', dmo_fn))
            self._data = {'z': hydro['z'], 'k': hydro['k'], 'Sk': hydro['Pk'] / dmo['Pk']}

        else:
            avail = self.available()
            raise ValueError(f"'{self.name}' not found. Available: {avail}")

        return self._data

    @staticmethod
    def _load_logpk(path):
        with open(path) as fh:
            header = fh.readline().strip().lstrip('#').split()
        data = np.genfromtxt(path, skip_header=1)
        logk = data[:, 0]
        k = 10.0 ** logk
        z_vals, Sk_cols = [], []
        for i, col in enumerate(header[1:], start=1):
            # column names: 'z100' -> z=1.00, 'logPk493' -> z=4.93
            if col.startswith('logPk'):
                z = float(col[5:]) / 100.0
            elif col.startswith('z'):
                z = float(col[1:]) / 100.0
            else:
                continue
            z_vals.append(z)
            Sk_cols.append(10.0 ** data[:, i])
        z_arr = np.array(z_vals)
        sort_idx = np.argsort(z_arr)
        return {
            'z':  z_arr[sort_idx],
            'k':  k,
            'Sk': np.array(Sk_cols)[sort_idx],
        }