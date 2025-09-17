import numpy as np 
import BCemu 

### Cosmology
Ob, Om = 0.0463, 0.2793

bcmdict = {'log10Mc': 13.32,
		   'mu'     : 0.93,
		   'thej'   : 4.235,  
		   'gamma'  : 2.25,
		   'delta'  : 6.40,
		   'eta'    : 0.15,
		   'deta'   : 0.14,
		   }
k_eval = 10**np.linspace(-1,1.08,50)

def test_BCM_7param():
	'''
	With this test, the 7 parameter baryonic power suppression is tested.
	'''
	bfcemu = BCemu.BCM_7param(Ob=Ob, Om=Om)
	p0     = bfcemu.get_boost(0.0, bcmdict, k_eval)
	p0p5   = bfcemu.get_boost(0.5, bcmdict, k_eval)
	p1     = bfcemu.get_boost(1.0, bcmdict, k_eval)
	p1p5   = bfcemu.get_boost(1.5, bcmdict, k_eval)
	p2     = bfcemu.get_boost(2.0, bcmdict, k_eval)
	assert np.abs(p0[0]-0.999129)<0.00001 and np.abs(p0p5[0]-0.998741)<0.00001 and np.abs(p1[0]-0.998928)<0.00001 and np.abs(p1p5[0]-0.999030)<0.00001 and np.abs(p2[0]-0.999575)<0.00001

def test_BCM_3param():
	'''
	With this test, the 3 parameter baryonic power suppression is tested.
	'''
	p0   = BCemu.get_boost_BCM_3param(0.0, bcmdict, k_eval, verbose=True)
	p0p5 = BCemu.get_boost_BCM_3param(0.5, bcmdict, k_eval, verbose=True)
	p1   = BCemu.get_boost_BCM_3param(1.0, bcmdict, k_eval, verbose=True)
	p1p5 = BCemu.get_boost_BCM_3param(1.5, bcmdict, k_eval, verbose=True)
	p2   = BCemu.get_boost_BCM_3param(2.0, bcmdict, k_eval, verbose=True)
	assert np.abs(p0[0]-0.999203)<0.00001 and np.abs(p0p5[0]-0.999118)<0.00001 and np.abs(p1[0]-0.999041)<0.00001 and np.abs(p1p5[0]-0.999229)<0.00001 and np.abs(p2[0]-0.999595)<0.00001
