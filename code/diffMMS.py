#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 
import dd as DD 
import mhfem_acc as mh 

from scipy.interpolate import interp1d 

''' find order of accurracy of LD and DD Eddington acceleration in the diffusion limit ''' 

def getError(N, solver):

	eps = 1e-9

	Sigmat = lambda x: 1/eps 
	Sigmaa = lambda x: .1*eps 

	n = 8 
	Q = np.ones((n,N-1))*eps 
	xb = 2
	x = np.linspace(0, xb, N)

	ld = solver(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	x, phi, it = ld.sourceIteration(1e-6)

	phi_f = interp1d(x, phi)

	# diffusion
	D = 1/(3*Sigmat(0))
	L = np.sqrt(D/Sigmaa(0))
	c2 = -eps/Sigmaa(0)/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	diff = lambda x: eps/Sigmaa(0) + c2*np.cosh(x/L)


	return np.fabs(phi_f(xb/2) - diff(xb/2))
	# return np.linalg.norm(phi - diff(x), 2)

def getOrder(solver):

	N = np.array([20, 40, 80, 160])
	err = np.zeros(len(N))

	for i in range(len(N)):

		err[i] = getError(N[i], solver)

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0])

	plt.loglog(1/N, err, '-o')

getOrder(DD.Eddington)
getOrder(LD.Eddington)

plt.show()