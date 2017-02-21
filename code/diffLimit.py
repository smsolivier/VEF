#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld
import dd 

import mhfem_acc as mh

''' compare LD and DD Eddington Acceleration in the Diffusion Limit (epsilon --> 0) ''' 

def getDiff(eps, solver):

	Sigmat = lambda x: 1/eps
	Sigmaa = lambda X: .1*eps

	N = 200
	n = 8 
	Q = np.ones((n,N))*eps
	xb = 2 
	x = np.linspace(0, xb, N)

	sn = solver(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	diff = mh.MHFEM(x, np.ones(N)/3, Sigmaa, Sigmat, np.ones(N), BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(1e-6)

	xd, phid = diff.solve(np.ones(N)*eps)

	assert (x == xd).all()

	return np.linalg.norm(phid - phi, 2), it

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
plt.legend(loc='best')
plt.show()