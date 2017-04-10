#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

sys.path.append('../../code')

import ld as LD 

from exactDiff import exactDiff

from hidespines import * 

''' compare the permuations of linear representation in diffusion limit ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1:] 
else:
	outfile = None 

Nruns = 10
eps = np.logspace(-6, 0, Nruns)

tol = 1e-12

def getIt(eps, opt, gauss):

	N = 100

	xb = 1 

	x0 = np.linspace(0, xb, N+1)
	Sigmat = lambda x: 1 
	Sigmaa = lambda x: .1 

	n = 8 

	it = np.zeros(len(eps))

	diff = np.zeros(len(eps))

	for i in range(len(eps)):

		sol = LD.Eddington(x0, n, lambda x: eps[i], lambda x: 1/eps[i], 
			np.ones((n,N))*eps[i], OPT=opt, GAUSS=gauss)

		x, phi, it[i] = sol.sourceIteration(tol, maxIter=200)

		phi_ex = exactDiff(eps[i], 1/eps[i], eps[i], xb)

		diff[i] = np.linalg.norm(phi - phi_ex(x), 2)/np.linalg.norm(phi_ex(x), 2)

	return diff, it 

diff00, it00 = getIt(eps, 0, 0)
diff10, it10 = getIt(eps, 1, 0)
diff01, it01 = getIt(eps, 0, 1)
diff11, it11 = getIt(eps, 1, 1)
diff20, it20 = getIt(eps, 2, 0)
diff21, it21 = getIt(eps, 2, 1)

plt.figure()
plt.loglog(eps, it00, '-o', label='No Slopes, No Gauss')
plt.loglog(eps, it10, '-o', label='Edge Slopes, No Gauss')
plt.loglog(eps, it01, '-o', clip_on=False, label='No Slopes, Gauss')
plt.loglog(eps, it11, '-o', clip_on=False, label='Edge Slopes, Gauss')
plt.loglog(eps, it20, '-o', label='vanLeer Centers, No Gauss')
plt.loglog(eps, it21, '-o', clip_on=False, label='vanLeer Centers, Gauss')

plt.xlabel(r'$\epsilon$', fontsize=18)
plt.ylabel('Number of Iterations', fontsize=18)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[0])

plt.figure()
plt.loglog(eps, diff00, '-o', label='No Slopes, No Gauss')
plt.loglog(eps, diff10, '-o', label='Edge Slopes, No Gauss')
plt.loglog(eps, diff01, '-o', clip_on=False, label='No Slopes, Gauss')
plt.loglog(eps, diff11, '-o', clip_on=False, label='Edge Slopes, Gauss')
plt.loglog(eps, diff20, '-o', label='vanLeer Centers, No Gauss')
plt.loglog(eps, diff21, '-o', clip_on=False, label='vanLeer, Gauss')

plt.xlabel(r'$\epsilon$', fontsize=18)
plt.ylabel('|Sn - Diffusion|', fontsize=18)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[1])
else:
	plt.show()