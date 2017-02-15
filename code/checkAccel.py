#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

from hidespines import * 

''' compare the number of iterations for S_2 with DSA and without ''' 

N = 101 # number of edges 
Sigmat = 1 
c = np.linspace(0, 1, 20)
Sigmaa = Sigmat*(1 - c) 
q = 1 
xb = 20

tol = 1e-6 

it = np.zeros(len(Sigmaa))
itmu = np.zeros(len(Sigmaa))
itdsa = np.zeros(len(Sigmaa))
tol = 1e-6

n = 8

x = np.linspace(0, xb, N)
q = np.ones(N)

for i in range(len(Sigmaa)):

	# solve Sn
	sn = Sn(x, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	x, phi, it[i] = sn.sourceIteration(tol)

	# solve mu 
	mu = muAccel(x, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	xmu, phimu, itmu[i] = mu.sourceIteration(tol)

	# solve dsa 
	dsa = DSA(x, n, lambda x: Sigmaa[i], lambda x: Sigmat, q)
	xdsa, phidsa, itdsa[i] = dsa.sourceIteration(tol)

print(it/itmu)
plt.figure(figsize=(8,6))
plt.plot(c, it, '-o', label='No Accel', clip_on=False)
plt.plot(c, itmu, '-*', label='Edd. Accel', clip_on=False)
plt.plot(c, itdsa, '->', label='DSA')
hidespines(plt.gca())
plt.yscale('log')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$\Sigma_s/\Sigma_t$', fontsize=20)
plt.ylabel('Number of Iterations', fontsize=16)
plt.savefig('../ans/accel.pdf')

plt.figure()
plt.plot(c, it/itmu)
plt.yscale('log')
plt.show()