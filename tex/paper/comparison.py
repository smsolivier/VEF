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

def compare(h, Sigmaa, Sigmat, Q, OPT, GAUSS, label):

	err_f = lambda a, b: np.linalg.norm(a - b, 2)/np.linalg.norm(b, 2)

	xb = 8
	N = np.array([int(xb/x) for x in h]) 

	print(N)

	err = np.zeros(len(N))

	n = 8 

	tol = 1e-10

	for i in range(len(N)):

		x = np.linspace(0, xb, N[i]+1)

		ld = LD.S2SA(x, n, Sigmaa, Sigmat, Q)
		ed = LD.Eddington(x, n, Sigmaa, Sigmat, Q, GAUSS=GAUSS, OPT=OPT)

		x, phi, it = ld.sourceIteration(tol, maxIter=1000)
		xe, phie, ite = ed.sourceIteration(tol)

		err[i] = err_f(phie, phi)

	fit = np.polyfit(np.log(xb/N), np.log(err), 1)

	print(fit[0])

	plt.loglog(xb/N, err, '-o', clip_on=False, label=label)

h = np.logspace(-2, -1, 3)

Sigmat = lambda x: 1 
Sigmaa = lambda x: .25 
Q = lambda x, mu: 1 

compare(h, Sigmaa, Sigmat, Q, 0, 0, '00')

Sigmamax = 50
Sigmat = lambda x: Sigmamax*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: Sigmamax*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
q = lambda x, mu: Sigmamax*(x<2) + 1*(x>=7)*(x<=8)

# compare(h, Sigmaa, Sigmat, q, 'Reed\'s Problem')

labelsize = 16 
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$h$', fontsize=labelsize)
plt.ylabel('S$_n$/VEF Convergence', fontsize=labelsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile)
else:
	plt.show()