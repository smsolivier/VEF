#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD 

Sigmat = lambda x: 50*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: 50*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
q = lambda x, mu: 50*(x<2) + 1*(x>=7)*(x<=8)

xb = 8

n = 8 
tol = 1e-10 

N = 750
x = np.linspace(0, xb, N+1)

ld = LD.LD(x, n, Sigmaa, Sigmat, q)
ed = LD.Eddington(x, n, Sigmaa, Sigmat, q)

x, phi, it = ld.sourceIteration(tol)
xe, phie, ite = ed.sourceIteration(tol)

plt.plot(x, phi, label='SI')
plt.plot(xe, phie, label='VEF')
plt.legend(loc='best')
plt.show()

