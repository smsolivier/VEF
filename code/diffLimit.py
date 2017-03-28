#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld
import dd 

from scipy.interpolate import interp1d

from hidespines import * 

''' compare LD and DD Eddington Acceleration in the Diffusion Limit (epsilon --> 0) ''' 

def getDiff(eps, solver):

	Sigmat = lambda x: 1/eps
	Sigmaa = lambda X: eps

	print('{:.6e}'.format(eps))

	N = 31
	n = 8 
	Q = np.ones((n,N-1))*eps
	xb = 1
	x = np.linspace(0, xb, N)

	tol = 1e-6

	sn = solver(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(tol, maxIter=100000)

	phi_f = interp1d(x, phi)

	# diffusion
	D = 1/(3*Sigmat(0))
	L = np.sqrt(D/Sigmaa(0))

	# print(L*xb/N)
	c2 = -eps/Sigmaa(0)/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	diff = lambda x: eps/Sigmaa(0) + c2*np.cosh(x/L)

	return np.linalg.norm(diff(x) - phi, 2), it
	# return np.fabs(phi_f(xb/2) - diff(xb/2)), it

N = 10
eps = np.logspace(-2, 0, N)

LD = np.zeros(N)
ED = np.zeros(N)
S2 = np.zeros(N)

itLD = np.zeros(N)
itED = np.zeros(N)
itS2 = np.zeros(N)

for i in range(N):

	LD[i], itLD[i] = getDiff(eps[i], ld.LD)
	ED[i], itED[i] = getDiff(eps[i], ld.Eddington)
	S2[i], itS2[i] = getDiff(eps[i], dd.S2SA)

print(itLD/itED)

plt.loglog(eps, itLD, '-o', label='Unaccelerated')
plt.loglog(eps, itED, '-*', label='Edd. Accelerated')
plt.loglog(eps, itS2, '->', label='S$_2$SA')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$\epsilon$', fontsize=20)
plt.ylabel('Number of Iterations', fontsize=20)
hidespines(plt.gca())
plt.savefig('../tex/figs/diffLimit.pdf', transparent=True)
plt.show()