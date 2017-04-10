#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 
import mhfem_acc as MH

from hidespines import * 

import sys 

''' compares diffusion and transport ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1:]
else:
	outfile = None 

N = 100
xb = 2
xe = np.linspace(0, xb, N+1)

n = 8 

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1

q = lambda x, mu: 1

BCL = 0 
BCR = 1 

tol = 1e-8 

ld = LD.LD(xe, n, Sigmaa, Sigmat, q, BCL, BCR) 

mh = MH.MHFEM(xe, Sigmaa, Sigmat, BCL, BCR, CENT=1)
mh.discretize(np.ones(N)/3, np.ones(N)/2)
xd, phid = mh.solve(np.ones(N), np.zeros(N))

x, phi, it = ld.sourceIteration(tol)

edd = ld.getEddington(.5*(ld.psiL + ld.psiR))
psiEdge = ld.edgePsi()
top = 0 
for i in range(n):

	top += np.fabs(ld.mu[i])*psiEdge[i,:] * ld.w[i] 

B = top/ld.zeroMoment(psiEdge)

mhc = MH.MHFEM(xe, Sigmaa, Sigmat, BCL, BCR, CENT=1)
mhc.discretize(edd, B)
xc, phic = mhc.solve(np.ones(N), np.zeros(N))

fsize = 20

plt.figure()
plt.plot(x*Sigmat(x), phi, label='S$_8$')
plt.plot(xd*Sigmat(xd), phid, label='Diffusion')
plt.xlabel(r'$\Sigma_t x$', fontsize=fsize)
plt.ylabel(r'$\phi(x)$', fontsize=fsize)
plt.legend(loc='best', frameon=False)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[0], transparent=True)

plt.figure()
plt.axhline(1/3, color='k', alpha=.4)
plt.plot(x*Sigmat(x), edd)
plt.xlabel(r'$\Sigma_t x$', fontsize=fsize)
plt.ylabel(r'$\langle \mu^2 \rightangle(x)$', fontsize=fsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[1], transparent=True)

plt.figure()
plt.plot(x*Sigmat(x), phi, '--', label='S$_8$')
plt.plot(x*Sigmat(x), phic, label='Corrected Diffusion')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$\Sigma_t x$', fontsize=fsize)
plt.ylabel(r'$\phi(x)$', fontsize=fsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[2], transparent=True)
else:
	plt.show()