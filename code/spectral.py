#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

N = 100 
n = 8 
xb = 2 

x = np.linspace(0, xb, N+1)

c = 1

Sigmat = lambda x: 1
Sigmaa = lambda x: Sigmat(x)*(1 - c)
Q = lambda x, mu: 1 

ld = LD.S2SA(x, n, Sigmaa, Sigmat, Q)

x, phi, it = ld.sourceIteration(1e-10)

rho = np.zeros(len(ld.diff) - 1)

for i in range(1, len(ld.diff)):

	rho[i-1] = ld.diff[i]/ld.diff[i-1]

plt.plot(rho/c)
plt.show()