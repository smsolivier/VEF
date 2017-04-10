#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

''' Test LD with Reed's Problem ''' 

N = 100
n = 8

Sigmat = lambda x: 3*(x>5) + .001*(x<=5)*(x>3) + 5*(x<=3)*(x>2) + 50*(x<=2)
Sigmaa = lambda x: .1*(x>5) + 5*(x<3)*(x>2) + 50*(x<=2)
Q = lambda x, mu: 1*(x<7)*(x>5) + 50*(x<=2)

xb = 9
tol = 1e-10
x = np.linspace(0, xb, N+1)

plt.semilogy(x, Sigmat(x), '--', label='Sigmat')
plt.semilogy(x, Sigmaa(x), label='Sigmaa')
# plt.semilogy(x, Q(x, 0), label='Q')
plt.legend(loc='best')
plt.show()

ld = LD.Eddington(x, n, Sigmaa, Sigmat, Q)

x, phi, it = ld.sourceIteration(tol)

plt.plot(x, phi)
plt.show()