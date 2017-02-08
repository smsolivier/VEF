#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

''' compare the number of iterations for S_2 with DSA and without ''' 

N = 500 # number of edges 
Sigmat = 1 
c = np.linspace(0, .95, 30)
Sigmaa = Sigmat*(1 - c) 
q = 1 
xb = 1 

tol = 1e-6 

it = np.zeros(len(Sigmaa))
itdsa = np.zeros(len(Sigmaa))
tol = 1e-6

for i in range(len(Sigmaa)):

	# solve Sn
	sn = Sn(N, Sigmaa[i], Sigmat, q, xb=20)
	x, phi, it[i] = sn.sourceIteration(tol)

	# solve DSA 
	dsa = DSA(N, Sigmaa[i], Sigmat, q, xb=20)
	xdsa, phidsa, itdsa[i] = dsa.sourceIteration(tol)

plt.plot(c, it, '-o', label=r'S$_2$')
plt.plot(c, itdsa, '-o', label=r'S$_2$ DSA')
plt.legend(loc='best')
plt.xlabel(r'$\Sigma_s/\Sigma_t$')
plt.ylabel('Number of Iterations')
plt.show()