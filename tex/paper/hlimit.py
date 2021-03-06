#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

from hidespines import * 

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

# xb = 1
# c = .9
# Sigmat = lambda x: 1 
# Sigmaa = lambda x: Sigmat(x)*(1 - c)
# q = lambda x, mu: 1

xb = 8 
Sigmat = lambda x: 50*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: 50*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
q = lambda x, mu: 50*(x<2) + 1*(x>=7)*(x<=8)

tol = 1e-10

h = np.logspace(-1.5, -1.1, 3)

N = np.array([int(xb/x) for x in h])

print(N)

ed00 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=0, GAUSS=0) for x in N]

ed01 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=0, GAUSS=1) for x in N]

ed10 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=1, GAUSS=0) for x in N]

ed11 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=1, GAUSS=1) for x in N]

ed20 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=2, GAUSS=0) for x in N]

ed21 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=2, GAUSS=1) for x in N]

diff00 = getDiff(ed00, tol)
# diff01 = getDiff(ed01, tol)
# diff10 = getDiff(ed10, tol)
# diff11 = getDiff(ed11, tol)
# diff20 = getDiff(ed20, tol)
diff21 = getDiff(ed21, tol)

# print(diff21/diff20)

fontsize=16
plt.loglog(xb/N, diff00, '-o', clip_on=False, label='None, Constant')
# plt.loglog(xb/N, diff01, '-v', clip_on=False, label='None, Linear')
# plt.loglog(xb/N, diff10, '-^', clip_on=False, label='Edge, Constant')
# plt.loglog(xb/N, diff11, '-<', clip_on=False, label='Edge, Linear')
# plt.loglog(xb/N, diff20, '->', clip_on=False, label='Center, Constant')
plt.loglog(xb/N, diff21, '-s', clip_on=False, label='Center, Linear')
plt.xlabel(r'$h$', fontsize=fontsize)
plt.ylabel('Convergence', fontsize=fontsize)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile, transparent=True)
else:
	plt.show()


