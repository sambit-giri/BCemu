import numpy as np 
import matplotlib.pyplot as plt
import BCemu

bfcemu = BCemu.BCM_7param(Ob=0.04, Om=0.30)
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

plt.semilogx(k_eval, p_eval, c='C0', lw=3, label='$f_b$={:.3f}'.format(bfcemu.fb))
plt.axis([1e-1,12,0.73,1.04])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)
plt.ylabel(r'$\frac{P_{\rm DM+baryon}}{P_{\rm DM}}$', fontsize=21)

bfcemu.update_cosmology(0.05, 0.30)
p_eval = bfcemu.get_boost(z, bcmdict, k_eval)
plt.semilogx(k_eval, p_eval, c='C1', lw=3, label='$f_b$={:.3f}'.format(bfcemu.fb))

bfcemu.update_cosmology(0.06, 0.30)
p_eval = bfcemu.get_boost(z, bcmdict, k_eval)
plt.semilogx(k_eval, p_eval, c='C2', lw=3, label='$f_b$={:.3f}'.format(bfcemu.fb))

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()