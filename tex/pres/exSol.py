#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

sys.path.append('../../code')

from hidespines import * 

import ld as LD 

if len(sys.argv) > 1:
	outfile = sys.argv[1] 
else:
	outfile = None

N = 100 
n = 8 
xb = 2 

x = np.linspace(0, xb, N+1) 

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1 

q = lambda x, mu: 1 

tol = 1e-10 

ld = LD.LD(x, n, Sigmaa, Sigmat, q)

x, phi, it = ld.sourceIteration(tol)

fontsize = 20 

plt.plot(x, phi)
plt.xlabel(r'$\Sigma_t x$', fontsize=fontsize)
plt.ylabel(r'$\phi(x)$', fontsize=fontsize)
hidespines(plt.gca())
if outfile != None: 
	plt.savefig(outfile, transparent=True)
else:
	plt.show()