#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld 

Neps = 30

eps = np.logspace(-6, 0, Neps)

Sigmat = 1/eps
Sigmaa = eps 
Q = eps

xb = 1

tol = 1e-6

N = np.array([50, 100, 200])
n = 8 

for j in range(len(N)):

	xe = np.linspace(0, xb, N[j])

	maxIter = 50

	it = np.zeros(Neps)

	for i in range(Neps):

		print('{:.6e} {}'.format(Sigmat[i], Sigmat[i]*xb/N[j]))

		sol = ld.Eddington(xe, n, lambda x: Sigmaa[i], lambda x: Sigmat[i], 
			np.ones((n,N[j]))*Q[i], BCL=0, BCR=1, CENT=1)

		# sol.setMMS()

		x, phi, it[i] = sol.sourceIteration(tol, maxIter=maxIter) 

	othick = Sigmat*xb/N[j]

	plt.loglog(othick, it, '-o', label=str(N[j]))

plt.legend(loc='best')
plt.show()
