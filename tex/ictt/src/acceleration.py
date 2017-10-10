#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from hidespines import * 

import sys 

sys.path.append('../../code')

import ld
import dd 

''' compare number of iterations in LD and DD Eddington Acceleration 
	in the Diffusion Limit (epsilon --> 0) 
''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

def getDiff(c, solver):

	Sigmat = lambda x: 1
	Sigmas = lambda x: c*Sigmat(0)
	Sigmaa = lambda x: Sigmat(x) - Sigmas(x)

	print('{}'.format(c))

	N = 50
	n = 8 
	Q = lambda x, mu: 1
	xb = 10
	x = np.linspace(0, xb, N+1)

	print('Sigmat*h =', Sigmat(0)*xb/N)

	tol = 1e-6

	sn = solver(x, n, Sigmaa, Sigmat, Q, BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(tol, maxIter=100000)

	phi_f = interp1d(x, phi)

	# diffusion
	D = 1/(3*Sigmat(0))
	L = np.sqrt(D/Sigmaa(0))

	# print(L*xb/N)
	c2 = -Q(0,0)/Sigmaa(0)/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	diff = lambda x: Q(0,0)/Sigmaa(0) + c2*np.cosh(x/L)

	return np.linalg.norm(diff(x) - phi, 2), it
	# return np.fabs(phi_f(xb/2) - diff(xb/2)), it

N = 20
c = np.linspace(0, 1, N)

LD = np.zeros(N)
ED = np.zeros(N)
S2 = np.zeros(N)

itLD = np.zeros(N)
itED = np.zeros(N)
itS2 = np.zeros(N)

for i in range(N):

	LD[i], itLD[i] = getDiff(c[i], ld.LD)
	ED[i], itED[i] = getDiff(c[i], ld.Eddington)
	S2[i], itS2[i] = getDiff(c[i], ld.S2SA)

print(itLD/itED)

colors = ['#3B7EA1', '#FDB515', '#ED4E33']

plt.semilogy(c, itLD, '-o', clip_on=False, label='SI', color=colors[0])
plt.semilogy(c, itED, '-*', clip_on=False, label='VEF', color=colors[1])
plt.semilogy(c, itS2, '->', clip_on=False, label='S$_2$SA', color=colors[2])
plt.legend(loc='best')
plt.xlabel(r'$\sigma_s/\sigma_t$')
plt.ylabel('Number of Iterations')
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile, transparent=True)
else:
	plt.show()