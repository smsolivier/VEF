#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 
import dd as DD 

from hidespines import * 

import sys 

''' compares difference between Sn and moment equations as cell width --> 0 ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

def getDiff(sol, tol=1e-6):

	diff = np.zeros(len(sol))
	for i in range(len(sol)):

		x, phi, it = sol[i].sourceIteration(tol)

		diff[i] = np.linalg.norm(phi - sol[i].phi_SN, 2)/np.linalg.norm(sol[i].phi_SN, 2)

	return diff 

N = 100 
n = 8 
xb = 1

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1 

tol = 1e-6 

N = np.logspace(1, 3, 5) 

for i in range(len(N)):

	N[i] = int(N[i])

ed00 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n, x)), OPT=0, GAUSS=0) for x in N]

ed01 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n, x)), OPT=0, GAUSS=1) for x in N]

ed10 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n, x)), OPT=1, GAUSS=0) for x in N]

ed11 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n, x)), OPT=1, GAUSS=1) for x in N]

diff00 = getDiff(ed00)
diff01 = getDiff(ed01)
diff10 = getDiff(ed10)
diff11 = getDiff(ed11)

fontsize=16
plt.loglog(xb/N, diff00, '-o', clip_on=False, label='00')
plt.loglog(xb/N, diff01, '-o', clip_on=False, label='01')
plt.loglog(xb/N, diff10, '-o', clip_on=False, label='10')
plt.loglog(xb/N, diff11, '-o', clip_on=False, label='11')
plt.xlabel(r'$h$', fontsize=fontsize)
plt.ylabel('SN/MHFEM Convergence', fontsize=fontsize)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile, transparent=True)
else:
	plt.show()


