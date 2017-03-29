#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import dd as DD 
import ld as LD 

from hidespines import * 

''' compare the number of iterations for unaccelerated, Eddington, and S2SA S8 ''' 

N = 100 # number of edges 
Sigmat = 1 
c = np.linspace(0, 1, 20)
Sigmaa = Sigmat*(1 - c) 
q = 1 
xb = 20

tol = 1e-6 

it = np.zeros(len(Sigmaa))
itmu = np.zeros(len(Sigmaa))
its2 = np.zeros(len(Sigmaa))

n = 8

x0 = np.linspace(0, xb, N+1)
q = np.ones((n,N))

maxIter = 10000

print('optical thickness =', xb/N*Sigmat)

for i in range(len(Sigmaa)):

	# solve Sn
	sn = LD.LD(x0, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	x, phi, it[i] = sn.sourceIteration(tol, maxIter)

	# solve mu 
	mu = LD.Eddington(x0, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	xmu, phimu, itmu[i] = mu.sourceIteration(tol, maxIter)

	# solve S2SA
	s2 = DD.S2SA(x0, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	xs2, phis2, its2[i] = s2.sourceIteration(tol, maxIter)

print(it/itmu)
plt.figure(figsize=(8,6))
plt.plot(c, it, '-o', label='No Accel', clip_on=False)
plt.plot(c, itmu, '-*', label='Edd. Accel', clip_on=False)
plt.plot(c, its2, '->', label='S$_2$SA')
hidespines(plt.gca())
plt.yscale('log')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$\Sigma_s/\Sigma_t$', fontsize=20)
plt.ylabel('Number of Iterations', fontsize=16)
plt.savefig('../tex/accel.pdf', transparent=True)

plt.figure()
plt.plot(c, it/itmu)
plt.yscale('log')
plt.show()