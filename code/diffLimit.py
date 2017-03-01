#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld
import dd 

from scipy.interpolate import interp1d

''' compare LD and DD Eddington Acceleration in the Diffusion Limit (epsilon --> 0) ''' 

def getDiff(eps, solver):

	Sigmat = lambda x: 1/eps
	Sigmaa = lambda X: .1*eps

	print('{:.6e}'.format(eps))

	N = 25
	n = 8 
	Q = np.ones((n,N))*eps
	xb = 1
	x = np.linspace(0, xb, N)

	tol = 1e-8

	sn = solver(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(tol)

	phi_f = interp1d(x, phi)

	# diffusion
	D = 1/(3*Sigmat(0))
	L = np.sqrt(D/Sigmaa(0))

	# print(L*xb/N)
	c2 = -eps/Sigmaa(0)/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	diff = lambda x: eps/Sigmaa(0) + c2*np.cosh(x/L)

	return np.linalg.norm(diff(x) - phi, 2), it
	# return np.fabs(phi_f(xb/2) - diff(xb/2)), it

N = 20
eps = np.logspace(-6, 0, N)

DD = np.zeros(N)
LD = np.zeros(N)

itDD = np.zeros(N)
itLD = np.zeros(N)

for i in range(N):

	LD[i], itLD[i] = getDiff(eps[i], ld.Eddington)
	DD[i], itDD[i] = getDiff(eps[i], dd.Eddington)

plt.figure()
plt.loglog(1/eps, DD, '-o', label='DD')
plt.loglog(1/eps, LD, '-o', label='LD')
plt.xlabel(r'$1/\epsilon$')
plt.ylabel('|| Sn - Diffusion ||')
plt.legend(loc='best')

plt.figure()
plt.plot(1/eps, itDD, '-o', label='DD')
plt.plot(1/eps, itLD, '-o', label='LD')
plt.xscale('log')
plt.legend(loc='best')
plt.show()