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

	err00 = np.zeros(nrun)
	err20 = np.zeros(nrun)
	err01 = np.zeros(nrun)
	err21 = np.zeros(nrun)

	for i in range(nrun):

		x = np.linspace(0, xb, N[i]+1)

		s2 = LD.S2SA(x, n, Sigmaa, Sigmat, Q)
		edd00 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=0, GAUSS=0)
		edd20 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=2, GAUSS=0)
		edd01 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=0, GAUSS=1)
		edd21 = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=2, GAUSS=1)

		x2, phi2, it2 = s2.sourceIteration(tol)
		x00, phi00, it00 = edd00.sourceIteration(tol)
		x20, phi20, it20 = edd20.sourceIteration(tol)
		x01, phi01, it01 = edd01.sourceIteration(tol)
		x21, phi21, it21 = edd21.sourceIteration(tol)

		err_f = lambda edd: np.linalg.norm(edd - phi2, 2)/np.linalg.norm(phi2, 2)

		err00[i] = err_f(phi00)
		err20[i] = err_f(phi20)
		err01[i] = err_f(phi01)
		err21[i] = err_f(phi21)

	print(err00/err20)
	print(err21/err20)

	fsize = 20
	plt.figure()
	plt.loglog(h, err00, '-o', clip_on=False, label='None, Const')
	plt.loglog(h, err01, '-o', clip_on=False, label='None, Linear')
	plt.loglog(h, err20, '-o', clip_on=False, label='van Leer, Const')
	plt.loglog(h, err21, '-o', clip_on=False, label='van Leer, Linear')
	plt.legend(loc='best', frameon=False)
	plt.xlabel('$h$', fontsize=fsize)
	plt.ylabel('SI/VEF Convergence', fontsize=fsize)
	hidespines(plt.gca())

nrun = 5 
h = np.logspace(-2, -1, nrun)
n = 8 
xb = 8
tol = 1e-6 

c = .75 
Sigmat = lambda x: 1 
Sigmaa = lambda x: Sigmat(x) * (1 - c) 
Q = lambda x, mu: 1 
makePlot(h, xb, n, Sigmaa, Sigmat, Q, tol)
if (outfile != None):
	plt.savefig(outfile[0])

h = np.logspace(-2, -1.1, nrun)
Sigmamax = 50
Sigmat = lambda x: Sigmamax*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: Sigmamax*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
Q = lambda x, mu: Sigmamax*(x<2) + 1*(x>=7)*(x<=8)
makePlot(h, xb, n, Sigmaa, Sigmat, Q, tol)
if (outfile != None):
	plt.savefig(outfile[1])
else:
	plt.show()
