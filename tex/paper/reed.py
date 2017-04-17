#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

Sigmat = lambda x: 100*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: 100*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
q = lambda x, mu: 100*(x<2) + 1*(x>=7)*(x<=8)

xb = 8

x = np.linspace(0, xb, 1000)
plt.figure()
plt.plot(x, Sigmat(x), '--', label=r'$\Sigma_t(x)$')
plt.figure()
plt.plot(x, Sigmaa(x), label=r'$\Sigma_a(x)$')
plt.figure()
plt.plot(x, q(x, 0), label=r'$Q(x)$')
# plt.legend(loc='best')
plt.show()

n = 8 
tol = 1e-10 

N = 250
x = np.linspace(0, xb, N+1)

ed = LD.Eddington(x, n, Sigmaa, Sigmat, q)

x, phi, it = ed.sourceIteration(tol)

plt.plot(x, phi)
plt.show()

