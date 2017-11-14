#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from hidespines import * 

import sys 
sys.path.append('../../code')

import ld as LD 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

N = 100
n = 8 
xb = 50

tol = 1e-6

nruns = 20 
eps = np.logspace(-6, 0, nruns)

ite = np.zeros(nruns)
its = np.zeros(nruns)

for i in range(nruns):

	Sigmaa = lambda x: eps[i]
	Sigmat = lambda x: 1/eps[i] 
	Q = lambda x, mu: eps[i] 

	x = np.linspace(0, xb, N+1)

	ed = LD.Eddington(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1, GAUSS=1, OPT=1)
	s2 = LD.S2SA(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	xe, phie, ite[i] = ed.sourceIteration(tol)
	xs, phis, its[i] = s2.sourceIteration(tol)

labelsize = 16
plt.semilogx(eps, ite, '-o', label='VEF', clip_on=False)
plt.semilogx(eps, its, '-*', label='S2SA', clip_on=False)
plt.xlabel(r'$\epsilon$', fontsize=labelsize)
plt.ylabel('Number of Iterations', fontsize=labelsize)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile)
else:
	plt.show()



