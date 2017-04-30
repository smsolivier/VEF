#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from hidespines import * 

import sys 

sys.path.append('../../code')

import ld as LD 

''' plot eddington and flux convergence for unaccelerated and accelerated Sn ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1:] 
else:
	outfile = None 

N = 100 

Sigmat = 1 
c = .75 
Sigmaa = Sigmat*(1 - c)
n = 8 
q = lambda x, mu: 1

xb = 20 

tol = 1e-6 

x = np.linspace(0, xb, N+1)

unac = LD.LD(x, n, lambda x: Sigmaa, lambda x: Sigmat, q)

acc = LD.Eddington(x, n, lambda x: Sigmaa, lambda x: Sigmat, q)

x, phi, it = unac.sourceIteration(tol)

xac, phiac, itac = acc.sourceIteration(tol)

fontsize = 20

# unaccelerated 
plt.figure()
plt.semilogy(np.arange(1, len(unac.phiConv)+1), 
	unac.phiConv, '-o', label=r'$\phi(x)$', clip_on=False)
plt.semilogy(np.arange(1, len(unac.eddConv)+1), 
	unac.eddConv, '-*', label=r'$\langle \mu^2 \rangle(x)$', clip_on=False)
plt.xlabel('Iteration Number', fontsize=fontsize)
plt.ylabel('Convergence', fontsize=fontsize)
plt.legend(loc='best', frameon=False, fontsize=fontsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[0], transparent=True)

# accelerated 
plt.figure()
plt.semilogy(np.arange(1, len(acc.phiConv)+1), 
	acc.phiConv, '-o', label=r'$\phi(x)$', clip_on=False)
plt.semilogy(np.arange(1, len(acc.eddConv)+1), 
	acc.eddConv, '-*', label=r'$\langle \mu^2 \rangle(x)$', clip_on=False)
plt.xlabel('Iteration Number', fontsize=fontsize)
plt.ylabel('Convergence', fontsize=fontsize)
plt.legend(loc='best', frameon=False, fontsize=fontsize)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile[1], transparent=True)
if (outfile == None):
	plt.show()