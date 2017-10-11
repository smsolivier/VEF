#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

from hidespines import * 

if (len(sys.argv) > 1):
	outfile = sys.argv[1:]
else:
	outfile = None 

def makePlot(h, xb, n, Sigmaa, Sigmat, Q, tol):

	nrun = len(h)

	N = np.array([int(xb/x) for x in h])

	err0 = np.zeros(nrun)
	err1 = np.zeros(nrun)

	for i in range(nrun):

		x = np.linspace(0, xb, N[i]+1)

		s2 = LD.S2SA(x, n, Sigmaa, Sigmat, Q)

		edd0 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=3, GAUSS=1)
		edd1 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=2, GAUSS=1)

		x2, phi2, it2 = s2.sourceIteration(tol)
		x0, phi0, it0 = edd0.sourceIteration(tol)
		x1, phi1, it1 = edd1.sourceIteration(tol)

		err_f = lambda edd: np.linalg.norm(edd - phi2, 2)/np.linalg.norm(phi2, 2)

		err0[i] = err_f(phi0)
		err1[i] = err_f(phi1)

	print(err0/err1)

	plt.figure()
	plt.loglog(h, err0, '-o', clip_on=False, label='Flat', color=colors[0])
	plt.loglog(h, err1, '-*', clip_on=False, label='Linear', color=colors[1])
	plt.xlim(h[0], h[-1])
	plt.legend()
	plt.xlabel('$h$')
	plt.ylabel(r'S$_N$/VEF Convergence')

colors = ['#3B7EA1', '#FDB515', '#ED4E33']

nrun = 5
h = np.logspace(-2, -1, nrun)
n = 8 
xb = 8
tol = 1e-6

c = .99
Sigmat = lambda x: 1 
Sigmaa = lambda x: Sigmat(x) * (1 - c) 
Q = lambda x, mu: 1 
makePlot(h, xb, n, Sigmaa, Sigmat, Q, tol)
plt.title('Homogeneous')
if (outfile != None):
	plt.savefig(outfile[0], transparent=True)

h = np.logspace(-2.3, -1.3, nrun)
Sigmamax = 10
Sigmat = lambda x: Sigmamax*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: Sigmamax*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
Q = lambda x, mu: Sigmamax*(x<2) + 1*(x>=7)*(x<=8)
makePlot(h, xb, n, Sigmaa, Sigmat, Q, tol)
plt.title('Reed\'s Problem')
if (outfile != None):
	plt.savefig(outfile[1], transparent=True)
else:
	plt.show()
