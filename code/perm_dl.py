#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

import sys 

''' compare the permuations of linear representation in diffusion limit ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

def getDiff(eps, sol, tol=1e-6):

	it = np.zeros(len(eps))

	for i in range(len(eps)):

		sol.Sigmat = lambda x: 1/eps[i] 

		sol.Sigmaa = lambda x: eps[i] 

		sol.q = np.ones((sol.n, sol.N))*eps[i] 

		x, phi, it[i] = sol.sourceIteration(tol)

	return it 

Nruns = 10 
eps = np.logspace(-6, 0, Nruns)

N = 30 

xb = 1 

x = np.linspace(0, xb, N+1)
Sigmat = lambda x: 1 
Sigmaa = lambda x: .1 

n = 8 
Q = np.ones((n, N))

tol = 1e-10 

ld00 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=0, GAUSS=0)
ld10 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=1, GAUSS=0)
ld01 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=0, GAUSS=1)
ld11 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=1, GAUSS=1)

it00 = getDiff(eps, ld00, tol)
it10 = getDiff(eps, ld10, tol)
it01 = getDiff(eps, ld01, tol)
it11 = getDiff(eps, ld11, tol)

plt.loglog(eps, it00, '-o', label='00')
plt.loglog(eps, it10, '-o', label='10')
plt.loglog(eps, it01, '-o', label='01')
plt.loglog(eps, it11, '-o', label='11')
plt.legend(loc='best')

plt.show()