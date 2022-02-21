"""
Created by Sambit K. Giri
"""

import numpy as np
from scipy import special
from scipy.interpolate import splev, splrep
import os, pickle, pkg_resources
from glob import glob

path_to_emu0_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z0_nComp3.pkl')
path_to_emu0p5_file = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z0p5_nComp3.pkl')
path_to_emu1_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z1_nComp3.pkl')
path_to_emu1p5_file = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z1p5_nComp3.pkl')
path_to_emu2_file   = pkg_resources.resource_filename('BCemu', 'input_data/kpls_emulator_z2_nComp3.pkl')

ks_emulated = np.array([ 0.0341045 ,  0.05861015,  0.08348237,  0.10855948,  0.13396836,
		        0.15800331,  0.18254227,  0.20724802,  0.23212444,  0.2567891 ,
		        0.28103674,  0.30569611,  0.33079354,  0.35528021,  0.37965142,
		        0.404148  ,  0.42867984,  0.45328332,  0.4779432 ,  0.50261366,
		        0.5271016 ,  0.55149765,  0.57618047,  0.60068947,  0.62521509,
		        0.64995821,  0.67464717,  0.69909405,  0.72353962,  0.74794006,
		        0.77257944,  0.79721068,  0.8218201 ,  0.84641959,  0.87094783,
		        0.89552692,  0.91995039,  0.94446984,  0.96905847,  0.99352717,
		        1.01807451,  1.04265491,  1.06732874,  1.09198076,  1.11652679,
		        1.14094596,  1.16538844,  1.18997249,  1.21454137,  1.23910819,
		        1.26372463,  1.28832777,  1.31294019,  1.33736488,  1.36188676,
		        1.38647099,  1.41088754,  1.43543816,  1.46000179,  1.48461337,
		        1.50917878,  1.5336324 ,  1.55828985,  1.58282587,  1.60737312,
		        1.63194377,  1.65639122,  1.68101056,  1.70559414,  1.73006237,
		        1.7545306 ,  1.77905407,  1.80374906,  1.82836316,  1.85291022,
		        1.87746965,  1.90194743,  1.9264312 ,  1.95096646,  1.97544012,
		        1.99995779,  2.02464604,  2.0492416 ,  2.07372144,  2.09825877,
		        2.12287081,  2.14742218,  2.17183929,  2.19641594,  2.22108696,
		        2.24561567,  2.28249413,  2.31919863,  2.34372827,  2.36822366,
		        2.39275106,  2.41736366,  2.45422376,  2.49100691,  2.51555959,
		        2.5401016 ,  2.57703286,  2.61373473,  2.63829902,  2.67525014,
		        2.71184415,  2.73638728,  2.77337491,  2.81011536,  2.83469182,
		        2.8716096 ,  2.90829837,  2.94512739,  2.98188263,  3.01883393,
		        3.05558917,  3.0923785 ,  3.12909011,  3.16603785,  3.21513235,
		        3.2519052 ,  3.28875968,  3.32549358,  3.36239263,  3.41145867,
		        3.44823639,  3.48510286,  3.5341483 ,  3.5832604 ,  3.62008455,
		        3.65703267,  3.7060856 ,  3.75515501,  3.80424913,  3.8532721 ,
		        3.902349  ,  3.95147159,  4.00055598,  4.04966883,  4.09877157,
		        4.14784735,  4.19693436,  4.24601388,  4.29511175,  4.34416169,
		        4.39321611,  4.45468133,  4.5159563 ,  4.56506615,  4.61413444,
		        4.67561052,  4.73691582,  4.78599272,  4.84740625,  4.90865463,
		        4.97012846,  5.03147234,  5.09286565,  5.15410242,  5.21558674,
		        5.27687181,  5.33827449,  5.41191905,  5.47320637,  5.53460643,
		        5.60825511,  5.68190978,  5.75552213,  5.82909965,  5.89035852,
		        5.9518537 ,  6.02549713,  6.09910125,  6.17273869,  6.25873037,
		        6.34454978,  6.41815464,  6.49181793,  6.5654902 ,  6.65144106,
		        6.73724886,  6.8108717 ,  6.89684204,  6.98270751,  7.06869507,
		        7.15448714,  7.24045448,  7.33862776,  7.42446926,  7.51047555,
		        7.60865781,  7.70682359,  7.79264862,  7.87861071,  7.97679073,
		        8.07497898,  8.17312454,  8.27126411,  8.38183955,  8.49219854,
		        8.59034035,  8.68854658,  8.79907933,  8.90941584,  9.01995908,
		        9.13035252,  9.24083808,  9.35119033,  9.46172083,  9.58445502,
		        9.70714802,  9.829853  ,  9.94022996, 10.05076646, 10.17348567,
		       10.30854414, 10.44348801, 10.56619074, 10.68889573, 10.82396168,
		       10.95882092, 11.09389811, 11.2288637 , 11.36392292, 11.51118508,
		       11.64610273, 11.78114023, 11.9283919 , 12.0756593 , 12.22293269,
		       12.37017986, 12.51740232])


def check_trained_emulators():
	if not ( len(glob(path_to_emu0_file))*len(glob(path_to_emu0p5_file))*len(glob(path_to_emu1_file))*len(glob(path_to_emu1p5_file))*len(glob(path_to_emu2_file)) ):
		from . import download
		download.download_emulators()

def ps_suppression_8param(theta, emul, return_std=False):
    log10Mc, mu, thej, gamma, delta, eta, deta, fb = theta
    # fb = 0.1612
    mins = np.array([11, 0.0, 2, 1, 3,  0.05, 0.05, 0.10])
    maxs = np.array([15, 2.0, 8, 4, 11, 0.40, 0.40, 0.25])
    assert mins[0]<=log10Mc<=maxs[0]
    assert mins[1]<=mu<=maxs[1]
    assert mins[2]<=thej<=maxs[2]
    assert mins[3]<=gamma<=maxs[3]
    assert mins[4]<=delta<=maxs[4]
    assert mins[5]<=eta<=maxs[5]
    assert mins[6]<=deta<=maxs[6]
    assert mins[7]<=fb<=maxs[7]
    theta = [log10Mc, mu, thej, gamma, delta, eta, deta, fb]
    if type(theta)==list: theta = np.array(theta)
    if theta.ndim==1: theta = theta[None,:]
    out = emul.predict_values(theta)#, return_std=True)
    # print(out.shape)
    return out.squeeze()

def nearest_element_idx(arr, a, both=True):
	if both:
		dist = np.abs(arr-a)
		dist_arg = np.argsort(dist)
		return  dist_arg[0], dist_arg[1]
	else:
		return np.abs(arr-a).argmin()

def clip_range(x, mn, mx, message=None):
	if x<mn: out = mn
	elif x>mx: out = mx
	else: out = x
	if out!=x:
		if message is not None: print('{} set from {:.3f} to {:.3f}'.format(message,x,out))
	return out 

class use_emul:
	def __init__(self, emul_names=None, Ob=0.0463, Om=0.2793, verbose=True):
		if emul_names is None:
			emul_names = {'0'  : path_to_emu0_file, 
						  '0.5': path_to_emu0p5_file,
						  '1'  : path_to_emu1_file,
						  '1.5': path_to_emu1p5_file,
						  '2'  : path_to_emu2_file
						  }
		self.emul_names = emul_names
		self.verbose    = verbose
		check_trained_emulators()
		self.load_emulators()
		self.update_cosmology(Ob, Om)
		self.ks0 = ks_emulated
		self.__dict__.update({'nu_Mc': 0, 'nu_mu': 0, 'nu_thej': 0, 'nu_gamma': 0, 'nu_delta': 0, 'nu_eta': 0, 'nu_deta': 0})

	def load_emulators(self, emul_names=None):
		if emul_names is not None: self.emul_names = emul_names
		emulators = []
		zs = []
		for ke in self.emul_names:
			zs.append(float(ke))
			emu = pickle.load(open(self.emul_names[ke],'rb'))
			emu.options['print_prediction'] = False
			emulators.append(emu)
		print('Emulators loaded.')
		self.emulators = np.array(emulators)
		self.emul_zs   = np.array(zs)

	def update_cosmology(self, Ob, Om):
		self.fb = Ob/Om
		if self.verbose: print('Baryon fraction is set to {:.3f}'.format(self.fb))

	def check_range(self):
		mins = [11, 0.0, 2, 1, 3,  0.05, 0.05, 0.10]
		maxs = [15, 2.0, 8, 4, 11, 0.40, 0.40, 0.25]
		self.mins = mins 
		self.maxs = maxs
		knob = False
		if mins[0]<=self.log10Mc<=maxs[0] and mins[1]<=self.mu<=maxs[1] and mins[2]<=self.thej<=maxs[2] and mins[3]<=self.gamma<=maxs[3] and mins[4]<=self.delta<=maxs[4] and mins[5]<=self.eta<=maxs[5] and mins[6]<=self.deta<=maxs[6] and mins[7]<=self.fb<=maxs[7]:
		    knob = True
		if not knob:
			print('The parameters provided are outside the allowed range.')
			print('log10Mc : [{},{}]'.format(self.mins[0],self.maxs[0]))
			print('mu      : [{},{}]'.format(self.mins[1],self.maxs[1]))
			print('thej    : [{},{}]'.format(self.mins[2],self.maxs[2]))
			print('gamma   : [{},{}]'.format(self.mins[3],self.maxs[3]))
			print('delta   : [{},{}]'.format(self.mins[4],self.maxs[4]))
			print('eta     : [{},{}]'.format(self.mins[5],self.maxs[5]))
			print('deta    : [{},{}]'.format(self.mins[6],self.maxs[6]))
			print('fb      : [{},{}]'.format(self.mins[7],self.maxs[7]))
		assert knob
		return None

	def z_evolve_param(self, z):
		self.log10Mc = clip_range(self.log10Mc*(1+z)**(-self.nu_Mc),  self.mins[0], self.maxs[0], message='log10Mc' if self.verbose else None)
		self.mu      = clip_range(self.mu*(1+z)**(-self.nu_mu),       self.mins[1], self.maxs[1], message='mu' if self.verbose else None)
		self.thej    = clip_range(self.thej*(1+z)**(-self.nu_thej),   self.mins[2], self.maxs[2], message='thej' if self.verbose else None)
		self.gamma   = clip_range(self.gamma*(1+z)**(-self.nu_gamma), self.mins[3], self.maxs[3], message='gamma' if self.verbose else None)
		self.delta   = clip_range(self.delta*(1+z)**(-self.nu_delta), self.mins[4], self.maxs[4], message='delta' if self.verbose else None)
		self.eta     = clip_range(self.eta*(1+z)**(-self.nu_eta),     self.mins[5], self.maxs[5], message='eta' if self.verbose else None)
		self.deta    = clip_range(self.deta*(1+z)**(-self.nu_deta),   self.mins[6], self.maxs[6], message='deta' if self.verbose else None)
		if self.verbose: 
			if not self.log10Mc or not self.mu or not self.thej or not self.gamma or not self.delta or not self.eta or not self.deta:
				if not z:
					print('Parameters evolved.')
		return None

	def make_theta(self, z):
		self.z_evolve_param(z)
		theta = [self.log10Mc, self.mu, self.thej, self.gamma, self.delta, self.eta, self.deta, self.fb]		
		return theta 

	def run(self, BCM_dict, z=0, nu_dict=None, return_std=False):
		assert 0<=z<=2
		self.BCM_dict = BCM_dict

		self.__dict__.update(BCM_dict)
		if nu_dict is not None: self.__dict__.update(nu_dict)

		self.check_range()

		if z in self.emul_zs:
			theta = self.make_theta(z)
			emu0  = self.emulators[self.emul_zs==z][0]
			ps = ps_suppression_8param(theta, emu0, return_std=return_std)
			if return_std: return ps[0], self.ks0, ps[1]
			else: return ps, self.ks0
		else:
			i0, i1 = nearest_element_idx(self.emul_zs, z)
			theta0 = self.make_theta(self.emul_zs[i0]) 
			emu0   = self.emulators[i0]
			theta1 = self.make_theta(self.emul_zs[i1]) 
			emu1   = self.emulators[i1]
			ps0 = ps_suppression_8param(theta0, emu0, return_std=return_std)
			ps1 = ps_suppression_8param(theta1, emu1, return_std=return_std)
			if return_std: return ps0[0] + (ps1[0]-ps0[0])*(z-self.emul_zs[i0])/(self.emul_zs[i1]-self.emul_zs[i0]), self.ks0, ps0[1] + (ps1[1]-ps0[1])*(z-self.emul_zs[i0])/(self.emul_zs[i1]-self.emul_zs[i0])
			else: return ps0 + (ps1-ps0)*(z-self.emul_zs[i0])/(self.emul_zs[i1]-self.emul_zs[i0]), self.ks0


class BCM_7param(use_emul):
	def __init__(self, emul_names=None, Ob=0.0463, Om=0.2793, verbose=True):
		super().__init__(emul_names=emul_names, Ob=Ob, Om=Om, verbose=verbose)

	def print_param_names(self):
		print('\nBaryonification parameters:')
		print('--------------------------')
		print('log10Mc, mu, thej, gamma, delta, eta, deta\n')
		print('Redshift evolution parameters:')
		print('-----------------------------')
		print('nu_Mc, nu_mu, nu_thej, nu_gamma, nu_delta, nu_eta, nu_deta\n')
	
	def get_boost(self, z, BCM_params, k_eval, fb=None):
		if k_eval.min()<self.ks0.min():
			print('k values below {:.3f} h/Mpc are errornous.'.format(self.ks0.min()))
		if k_eval.max()>self.ks0.max():
			print('k values above {:.3f} h/Mpc are errornous.'.format(self.ks0.max()))
		if fb is not None: self.fb = fb
		pp, kk = self.run(BCM_params, z=z)
		pp_tck = splrep(kk, pp, k=3)
		return splev(k_eval, pp_tck, ext=0)


class BCM_3param(use_emul):
	def __init__(self, emul_names=None, Ob=0.0463, Om=0.2793, verbose=True):
		super().__init__(emul_names=emul_names, Ob=Ob, Om=Om, verbose=verbose)

	def print_param_names(self):
		print('\nBaryonification parameters:')
		print('--------------------------')
		print('log10Mc, thej, deta\n')
		print('Redshift evolution parameters:')
		print('-----------------------------')
		print('nu_Mc, nu_thej, nu_deta\n')
	
	def get_boost(self, z, BCM_params, k_eval, fb=None):
		BCM_params['delta'] = 7.0
		BCM_params['eta']   = 0.2
		BCM_params['mu']    = 1.0
		BCM_params['gamma'] = 2.5
		if k_eval.min()<self.ks0.min():
			print('k values below {:.3f} h/Mpc are errornous.'.format(self.ks0.min()))
		if k_eval.max()>self.ks0.max():
			print('k values above {:.3f} h/Mpc are errornous.'.format(self.ks0.max()))
		if fb is not None: self.fb = fb
		pp, kk = self.run(BCM_params, z=z)
		pp_tck = splrep(kk, pp, k=3)
		return splev(k_eval, pp_tck, ext=0)


