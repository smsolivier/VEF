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

Nruns = 15
eps = np.logspace(-3, 0, Nruns)

tol = 1e-6

def getIt(eps, opt, gauss):

	N = 50

	xb = 10

	x0 = np.linspace(0, xb, N+1)
	Sigmat = lambda x: 1 
	Sigmaa = lambda x: .1 
	q = lambda x: 1

	n = 8 

	it = np.zeros(len(eps))

	diff = np.zeros(len(eps))

	for i in range(len(eps)):

		sol = LD.Eddington(x0, n, lambda x: eps[i], lambda x: 1/eps[i], 
			lambda x, mu: eps[i], OPT=opt, GAUSS=gauss)

		x, phi, it[i] = sol.sourceIteration(tol, maxIter=200)

		phi_ex = exactDiff(eps[i], 1/eps[i], eps[i], xb)

		diff[i] = np.linalg.norm(phi - phi_ex(x), 2)/np.linalg.norm(phi_ex(x), 2)

	return diff, it 

diff0, it0 = getIt(eps, 3, 1)
diff1, it1 = getIt(eps, 2, 1)
# diff01, it01 = getIt(eps, 0, 1)
# diff11, it11 = getIt(eps, 1, 1)
# diff20, it20 = getIt(eps, 2, 0)
# diff21, it21 = getIt(eps, 2, 1)

plt.figure()
plt.semilogx(eps, it0, '-o', clip_on=False, label='Flat')
plt.semilogx(eps, it1, '-*', clip_on=False, label='Linear')

plt.xlabel(r'$\epsilon$')
plt.ylabel('Number of Iterations')
plt.legend()
if (outfile != None):
	plt.savefig(outfile[0])

plt.figure()
plt.loglog(eps, diff0, '-o', clip_on=False, label='Flat')
plt.loglog(eps, diff1, '-*', clip_on=False, label='Linear')

plt.xlabel(r'$\epsilon$')
plt.ylabel('VEF/Exact Diffusion Error')
plt.legend()
if (outfile != None):
	plt.savefig(outfile[1])
else:
	plt.show()