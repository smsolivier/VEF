#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD 

''' Plot error from MMS solution for LD ''' 

N = 100
n = 8 
xb = 1
x = np.linspace(0, xb, N+1)

Sigmaa = lambda x: .1*(x < xb/2) 
Sigmat = lambda x: 1

q = lambda x, mu: 1

tol = 1e-12

ld = LD.LD(x, n, Sigmaa, Sigmat, q)
ld.setMMS()
ed = LD.Eddington(x, n, Sigmaa, Sigmat, q)
ed.setMMS()

x, phi, it = ld.sourceIteration(tol)
xe, phie, ite = ed.sourceIteration(tol)

diff = np.fabs(phi - ld.phi_mms(x))/ld.phi_mms(x)
diffe = np.fabs(phi - ed.phi_mms(xe))/ed.phi_mms(xe)
plt.semilogy(x, diff, '-o')
plt.semilogy(xe, diffe, '-o')
plt.show()