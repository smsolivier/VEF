#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

from hidespines import * 

N = 101
Sigmat = 1 
c = .75
Sigmaa = Sigmat*(1 - c) 
q = 1 
xb = 20 

tol = 1e-6 

n = 8 

sn = Sn(np.linspace(0, xb, N), n, lambda x: Sigmaa, lambda x: Sigmat, np.ones(N)*q)
mu = muAccel(np.linspace(0, xb, N), n, lambda x: Sigmaa, lambda x: Sigmat, np.ones(N)*q)

x, phi, it = sn.sourceIteration(tol)
xmu, phimu, itmu = mu.sourceIteration(tol)

plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(sn.phiCon)+1), sn.phiCon, '-o', label=r'$\phi(x)$', clip_on=False)
plt.plot(np.arange(1, len(sn.eddCon)+1), sn.eddCon, '-*', label=r'$\langle \mu^2 \rangle(x)$', clip_on=False)
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Convergence')
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
plt.savefig('../ans/eddCon_si.pdf')

plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(mu.phiCon)+1), mu.phiCon, '-o', label=r'$\phi(x)$', clip_on=False)
plt.plot(np.arange(1, len(mu.eddCon)+1), mu.eddCon, '-*', label=r'$\langle \mu^2 \rangle(x)$', clip_on=False)
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Convergence')
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
plt.savefig('../ans/eddCon_mu.pdf')

plt.show()

