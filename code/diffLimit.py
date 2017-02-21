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

	N = 250
	n = 8 
	Q = np.ones(N)*eps
	xb = 2 
	x = np.linspace(0, xb, N)

	# sn = ld.LD(x, n, Sigmaa, Sigmat, Q)
	sn = solver(x, n, Sigmaa, Sigmat, Q)

	diff = mh.MHFEM(x, np.ones(N)/3, Sigmaa, Sigmat, np.ones(N), BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(1e-6)

	xd, phid = diff.solve(Q)

	assert (x == xd).all()

	return np.linalg.norm(phid - phi, 2)

N = 20
eps = np.logspace(-6, 0, N)

DD = np.zeros(N)
LD = np.zeros(N)
for i in range(N):

	DD[i] = getDiff(eps[i], dd.Eddington)
	LD[i] = getDiff(eps[i], ld.LD)

plt.loglog(1/eps, DD, '-o', label='DD')
plt.loglog(1/eps, LD, '-o', label='LD')
plt.xlabel(r'$1/\epsilon$')
plt.ylabel('|| Sn - Diffusion ||')
plt.legend(loc='best')
plt.show()