import numpy as np 
import pickle
from BCemu import * 

### Cosmology
Ob, Om = 0.0463, 0.2793
bfcemu = BCemu2025()

bcmdict = {'log10Mc': 13.32,
		   'mu'     : 0.93,
		   'thej'   : 4.235,  
		   'gamma'  : 2.25,
		   'delta'  : 6.40,
		   'eta'    : 0.15,
		   'deta'   : 0.14,
		   }
k_eval = 10**np.linspace(-1,1.08,50)
p0     = bfcemu.get_boost(0.0, bcmdict, k_eval)
p0p5   = bfcemu.get_boost(0.5, bcmdict, k_eval)
p1     = bfcemu.get_boost(1.0, bcmdict, k_eval)
p1p5   = bfcemu.get_boost(1.5, bcmdict, k_eval)
p2     = bfcemu.get_boost(2.0, bcmdict, k_eval)

# Read the BAHAMAS data
BAH = pickle.load(open('BAHAMAS_data.pkl', 'rb'))

# Plot

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['axes.labelsize']  = 20
rcParams['font.size']       = 20 
rcParams['axes.linewidth']  = 1.6

plt.figure(figsize=(15,9))

ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

ax1.set_title('$z=0$')
ax1.semilogx(BAH['z=0']['k'], BAH['z=0']['S'], '-', c='C0', lw=5, alpha=0.2, label='BAHAMAS')
ax1.semilogx(k_eval, p0, '--', c='b', lw=3, label='Emulated')

ax1.set_xlim(0.09,12)
ax1.set_ylim(0.7,1.08)
ax1.legend()
ax1.set_xlabel('$k$ ($h$ Mpc)')
ax1.set_ylabel('$\mathcal{S}(k)$')

ax2.set_title('$z=0.5$')
ax2.semilogx(BAH['z=0.5']['k'], BAH['z=0.5']['S'], '-', c='C0', lw=5, alpha=0.2)
ax2.semilogx(k_eval, p0p5, '--', c='b', lw=3, label='Emulated')

ax2.set_xlim(0.09,12)
ax2.set_ylim(0.7,1.08)
#ax2.legend()
ax2.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
ax2.set_ylabel('$\mathcal{S}(k)$')

ax3.set_title('$z=1$')
ax3.semilogx(BAH['z=1']['k'], BAH['z=1']['S'], '-', c='C0', lw=5, alpha=0.2)
ax3.semilogx(k_eval, p1, '--', c='b', lw=3, label='Emulated')

ax3.set_xlim(0.09,12)
ax3.set_ylim(0.7,1.08)
#ax3.set_legend()
ax3.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
ax3.set_ylabel('$\mathcal{S}(k)$')

ax4.set_title('$z=1.5$')
ax4.semilogx(BAH['z=1.5']['k'], BAH['z=1.5']['S'], '-', c='C0', lw=5, alpha=0.2)
ax4.semilogx(k_eval, p1p5, '--', c='b', lw=3, label='Emulated')

ax4.set_xlim(0.09,12)
ax4.set_ylim(0.7,1.08)
#ax4.legend()
ax4.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
ax4.set_ylabel('$\mathcal{S}(k)$')

ax5.set_title('$z=2$')
ax5.semilogx(BAH['z=2']['k'], BAH['z=2']['S'], '-', c='C0', lw=5, alpha=0.2)
ax5.semilogx(k_eval, p2, '--', c='b', lw=3, label='Emulated')

ax5.set_xlim(0.09,12)
ax5.set_ylim(0.7,1.08)
#ax5.legend()
ax5.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
ax5.set_ylabel('$\mathcal{S}(k)$')

# plt.tight_layout()
# plt.show()



bcmdict = {'log10Mc': 13.32,
		   'mu'     : 0.93,
		   'thej'   : 4.235,  
		   'gamma'  : 2.25,
		   'delta'  : 6.40,
		   'eta'    : 0.15,
		   'deta'   : 0.14,
		   'nu_Mc'  : 0.05,
		   }
k_eval = 10**np.linspace(-1,1.08,50)
p0     = bfcemu.get_boost(0.0, bcmdict, k_eval)
p0p5   = bfcemu.get_boost(0.5, bcmdict, k_eval)
p1     = bfcemu.get_boost(1.0, bcmdict, k_eval)
p1p5   = bfcemu.get_boost(1.5, bcmdict, k_eval)
p2     = bfcemu.get_boost(2.0, bcmdict, k_eval)


ax1.semilogx(k_eval, p0, '-.', c='g', lw=3, label='Emulated')
ax2.semilogx(k_eval, p0p5, '-.', c='g', lw=3, label='Emulated')
ax3.semilogx(k_eval, p1, '-.', c='g', lw=3, label='Emulated')
ax4.semilogx(k_eval, p1p5, '-.', c='g', lw=3, label='Emulated')
ax5.semilogx(k_eval, p2, '-.', c='g', lw=3, label='Emulated')
plt.tight_layout()
plt.show()
