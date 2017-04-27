#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

from hidespines import * 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

def compare(h, Sigmaa, Sigmat, Q, label):

	err_f = lambda a, b: np.linalg.norm(a - b, 2)/np.linalg.norm(b, 2)

	xb = 8
	N = np.array([Sigmat(0)*int(xb/x) for x in h]) 

	print(N)

	err = np.zeros(len(N))

	n = 8 

	tol = 1e-10

	for i in range(len(N)):

		x = np.linspace(0, xb, N[i]+1)

		ld = LD.LD(x, n, Sigmaa, Sigmat, Q)
		ed = LD.Eddington(x, n, Sigmaa, Sigmat, Q, GAUSS=1, OPT=2)

		x, phi, it = ld.sourceIteration(tol, maxIter=1000)
		xe, phie, ite = ed.sourceIteration(tol)

		err[i] = err_f(phie, phi)

	plt.loglog(Sigmat(0)*xb/N, err, '-o', clip_on=False, label=label)

h = np.logspace(-.2, -.05, 3)

Sigmat = lambda x: 1 
Sigmaa = lambda x: .25 
Q = lambda x, mu: 1 

compare(h, Sigmaa, Sigmat, Q, 'Homogeneous Slab')

Sigmat = lambda x: 50*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: 50*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
q = lambda x, mu: 50*(x<2) + 1*(x>=7)*(x<=8)

compare(h, Sigmaa, Sigmat, q, 'Reed\'s Problem')

labelsize = 16 
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$h$', fontsize=labelsize)
plt.ylabel('S$_n$/VEF Convergence', fontsize=labelsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile)
else:
	plt.show()