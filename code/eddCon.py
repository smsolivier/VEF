#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

N = 100 
sn = muAccel(np.linspace(0, 10, N), 8, lambda x: .1, lambda x: .83, np.ones(N))

x, phi, it = sn.sourceIteration(1e-6)

# plt.plot(np.arange(1, len(sn.phiCon)+1), sn.phiCon, '-o', label=r'$\phi$')
# plt.plot(np.arange(1, len(sn.eddCon)+1), sn.eddCon, '-o', label=r'$\langle \mu^2 \rangle$')
# plt.yscale('log')
# plt.legend(loc='best')
plt.plot(np.array(sn.phiCon)/np.array(sn.eddCon))
plt.show()

