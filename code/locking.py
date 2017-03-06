#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import dd 

Neps = 20 

c = np.linspace(0, 1, Neps)

Sigmat = np.logspace(0, 6, Neps)
Sigmaa = 0 
Q = 100

xb = 1

tol = 1e-6

N = 50
n = 8 

PLOT = np.zeros(Neps)
PLOT[11] = 1 

xe = np.linspace(0, xb, N)

maxIter = 50

it = np.zeros(Neps)

for i in range(Neps):

	print('{:.6e} {}'.format(Sigmat[i], Sigmat[i]*xb/N))

	sol = dd.Eddington(xe, n, lambda x: Sigmaa, lambda x: Sigmat[i], 
		np.ones((n,N))*Q, BCL=0, BCR=1, CENT=1)

	# sol.setMMS()

	x, phi, it[i] = sol.sourceIteration(tol, maxIter=maxIter, PLOT=PLOT[i]) 

	if (it[i] == maxIter):

		plt.plot(x, phi, label=str(Sigmat[i]))

	# else:

	# 	plt.plot(x, phi, '-o', color='b')

othick = Sigmat*xb/N

plt.legend(loc='best')
plt.show()
plt.loglog(othick, it, '-o')
plt.show()
