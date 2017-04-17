#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

Sigmat = lambda x: 3*(x>5) + .001*(x<=5)*(x>3) + 5*(x<=3)*(x>2) + 50*(x<=2)
Sigmaa = lambda x: .1*(x>5) + 5*(x<=3)*(x>2) + 50*(x<=2)
Q = lambda x, mu: 1*(x<7)*(x>5) + 50*(x<=2)

xb = 9

x = np.linspace(0, xb, 1000)
plt.figure()
plt.plot(x, Sigmat(x), '--', label=r'$\Sigma_t(x)$')
plt.figure()
plt.plot(x, Sigmaa(x), label=r'$\Sigma_a(x)$')
plt.figure()
plt.plot(x, Q(x, 0), label=r'$Q(x)$')
# plt.legend(loc='best')
plt.show()

tol = 1e-10

n = 8 

N = np.array([25, 50, 100, 200])

erre = np.zeros(len(N))
err2 = np.zeros(len(N))

err_f = lambda a, b: np.linalg.norm(a - b, 2)/np.linalg.norm(b, 2)

for i in range(len(N)):

	x = np.linspace(0, xb, N[i]+1)

	ld = LD.LD(x, n, Sigmaa, Sigmat, Q)
	ed = LD.Eddington(x, n, Sigmaa, Sigmat, Q, GAUSS=0, OPT=2)
	s2 = LD.S2SA(x, n, Sigmaa, Sigmat, Q)

	x, phi, it = ld.sourceIteration(tol, maxIter=1000)

	xe, phie, ite = ed.sourceIteration(tol)

	x2, phi2, it2 = s2.sourceIteration(tol)

	erre[i] = err_f(phie, phi)
	err2[i] = err_f(phi2, phi)

plt.loglog(xb/N, erre, '-o', label='VEF')
plt.loglog(xb/N, err2, '-o', label='S2SA')
plt.legend(loc='best')
plt.show()

