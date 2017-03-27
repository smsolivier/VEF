#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

from hidespines import * 

N = 100 
n = 8 
xb = 1

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1 

tol = 1e-6 

N = np.logspace(1, 3, 5) 

for i in range(len(N)):

	N[i] = int(N[i])

diff = np.zeros(len(N))

for i in range(len(N)):

	xe = np.linspace(0, xb, N[i]+1) 

	q = np.ones((n,N[i]))

	ed = LD.Eddington(xe, n, Sigmaa, Sigmat, q)

	x, phi, it = ed.sourceIteration(tol) 

	diff[i] = np.linalg.norm(phi - ed.phi_SN, 2) / np.linalg.norm(ed.phi_SN, 2)

plt.loglog(xb/N, diff, '-o', clip_on=False)
plt.xlabel(r'$h$', fontsize=20)
plt.ylabel('Convergence', fontsize=20)
hidespines(plt.gca())
plt.savefig('../tex/figs/hlim.pdf', transparent=True)
plt.show()


