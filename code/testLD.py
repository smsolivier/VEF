#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

N = 100
n = 8 
xb = 1
x = np.linspace(0, xb, N+1)

Sigmaa = lambda x: .1 
Sigmat = lambda x: 1 

q = np.ones((n, N))

tol = 1e-12

ld = LD.LD(x, n, Sigmaa, Sigmat, q)
ld.setMMS()

x, phi, it = ld.sourceIteration(tol)

diff = np.fabs(phi - ld.phi_mms(x))/ld.phi_mms(x)
plt.semilogy(x, diff, '-o')
plt.show()