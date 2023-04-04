import numpy as np 
import matplotlib.pyplot as plt
import BCemu
from tqdm import tqdm

Ob, Om = 0.0463, 0.2793

bfcemu = BCemu.BCM_7param(Ob=Ob, Om=Om)
bcmdict = {'log10Mc': 13.32,
           'mu'     : 0.93,
           'thej'   : 4.235,  
           'gamma'  : 2.25,
           'delta'  : 6.40,
           'eta'    : 0.15,
           'deta'   : 0.14,
           }

z = 0
k_eval = 10**np.linspace(-1,1.08,50)
p_eval = bfcemu.get_boost(z, bcmdict, k_eval)

plt.semilogx(k_eval, p_eval, c='C0', lw=3)
plt.axis([1e-1,12,0.73,1.04])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)
plt.ylabel(r'$\frac{P_{\rm DM+baryon}}{P_{\rm DM}}$', fontsize=21)
plt.tight_layout()
plt.show()

n_samples = 1000
mins = [11, 0.0, 2, 1, 3,  0.05, 0.05, 0.10]
maxs = [15, 2.0, 8, 4, 11, 0.40, 0.40, 0.25]

pses1 = []
for i in tqdm(range(n_samples)):
    bfcemu = BCemu.BCM_7param(Ob=Ob, Om=Om)
    bcmdict = {'log10Mc': np.random.uniform(mins[0], maxs[0], 1)[0],
               'mu'     : np.random.uniform(mins[1], maxs[1], 1)[0],
               'thej'   : np.random.uniform(mins[2], maxs[2], 1)[0],  
               'gamma'  : np.random.uniform(mins[3], maxs[3], 1)[0],
               'delta'  : np.random.uniform(mins[4], maxs[4], 1)[0],
               'eta'    : np.random.uniform(mins[5], maxs[5], 1)[0],
               'deta'   : np.random.uniform(mins[6], maxs[6], 1)[0],
               }
    p_eval = bfcemu.get_boost(z, bcmdict, k_eval)
    pses1.append(p_eval)
pses1 = np.array(pses1)


# plt.semilogx(k_eval, p_eval, c='C0', lw=3)
plt.fill_between(k_eval, pses1.min(axis=0), pses1.max(axis=0), color='C1', alpha=0.3, label='Planck cosmology')
plt.axis([1e-1,12,0.3,1.6])
plt.xscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)
plt.ylabel(r'$\frac{P_{\rm DM+baryon}}{P_{\rm DM}}$', fontsize=21)
plt.tight_layout()
plt.show()

pses2 = []
for i in tqdm(range(n_samples)):
    bfcemu = BCemu.BCM_7param(Ob=Om*np.random.uniform(mins[7], maxs[7], 1)[0], Om=Om)
    bcmdict = {'log10Mc': np.random.uniform(mins[0], maxs[0], 1)[0],
               'mu'     : np.random.uniform(mins[1], maxs[1], 1)[0],
               'thej'   : np.random.uniform(mins[2], maxs[2], 1)[0],  
               'gamma'  : np.random.uniform(mins[3], maxs[3], 1)[0],
               'delta'  : np.random.uniform(mins[4], maxs[4], 1)[0],
               'eta'    : np.random.uniform(mins[5], maxs[5], 1)[0],
               'deta'   : np.random.uniform(mins[6], maxs[6], 1)[0],
               }
    p_eval = bfcemu.get_boost(z, bcmdict, k_eval)
    pses2.append(p_eval)
pses2 = np.array(pses2)


# plt.semilogx(k_eval, p_eval, c='C0', lw=3)
plt.fill_between(k_eval, pses2.min(axis=0), pses2.max(axis=0), color='k', alpha=0.3, label='Cosmology left free')
plt.fill_between(k_eval, pses1.min(axis=0), pses1.max(axis=0), color='C1', alpha=0.3, label='Planck cosmology')
plt.axis([1e-1,12,0.3,1.6])
plt.xscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)
plt.ylabel(r'$\frac{P_{\rm DM+baryon}}{P_{\rm DM}}$', fontsize=21)
plt.legend()
plt.tight_layout()
plt.show()




