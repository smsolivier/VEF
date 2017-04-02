#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from hidespines import * 

import sys 

sys.path.append('../../code')

import ld as LD
''' compares difference between Sn and moment equations as cell width --> 0 ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

N = 100 
n = 8 
xb = 1

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1 

tol = 1e-6 

N = np.logspace(1, 4, 5) 

for i in range(len(N)):

	N[i] = int(N[i])

diff = np.zeros(len(N))

for i in range(len(N)):

	xe = np.linspace(0, xb, N[i]+1) 

	q = np.ones((n,N[i]))

	ed = LD.Eddington(xe, n, Sigmaa, Sigmat, q, OPT=0)

	x, phi, it = ed.sourceIteration(tol)

	diff[i] = np.linalg.norm(phi - ed.phi_SN, 2)/np.linalg.norm(ed.phi_SN, 2)

fontsize=16
plt.loglog(xb/N, diff, '-o', clip_on=False)
plt.xlabel(r'$h$', fontsize=fontsize)
plt.ylabel('SN/MHFEM Convergence', fontsize=fontsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile, transparent=True)
else:
	plt.show()


