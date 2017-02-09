#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

from hidespines import * 

''' compare the number of iterations for S_2 with DSA and without ''' 

N = 100 # number of edges 
Sigmat = 1 
c = np.linspace(0, .95, 20)
Sigmaa = Sigmat*(1 - c) 
q = 1 
xb = 1 

tol = 1e-6 

it = np.zeros(len(Sigmaa))
itmu = np.zeros(len(Sigmaa))
itdsa = np.zeros(len(Sigmaa))
tol = 1e-6

n = 8

for i in range(len(Sigmaa)):

	# solve Sn
	sn = Sn(N, n, Sigmaa[i], Sigmat, q, xb=20)
	x, phi, it[i] = sn.sourceIteration(tol)

	# solve mu 
	mu = muAccel(N, n, Sigmaa[i], Sigmat, q, xb=20)
	xmu, phimu, itmu[i] = mu.sourceIteration(tol)

	# solve dsa 
	dsa = DSA(N, n, Sigmaa[i], Sigmat, q, xb=20)
	xdsa, phidsa, itdsa[i] = dsa.sourceIteration(tol)

plt.plot(c, it, '-o', label='S$_8$ No Accel')
plt.plot(c, itmu, '-o', label='S$_8$ Edd. Accel')
plt.plot(c, itdsa, '-o', label='S$_8$ DSA')
hidespines(plt.gca())
plt.yscale('log')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$\Sigma_s/\Sigma_t$')
plt.ylabel('Number of Iterations')
plt.savefig('accel.pdf')
plt.show()