#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fv_diff import * 
from mhfem_diff import * 

Sigmaa = .1 
Sigmat = .83 

Q = 1 

D = 1/(3*Sigmat) 
L = np.sqrt(D/Sigmaa)

xb = 1 

N = 250
BC = 1

# mhfem 
mhfem = MHFEM(N, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2)
xfe, phife = mhfem.solve(np.ones(N)*Q)

# fv 
fv = finiteVolume(N, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2)
xfv, phifv = fv.solve(np.ones(N)*Q)

# exact solution 
atilde = xb + 2*D 
x_ex = np.linspace(0, xb, 50)
phi_ex = Q/Sigmaa*(1 - np.cosh(x_ex/L)/np.cosh(atilde/L))

plt.plot(xfe, phife, label='MHFEM')
plt.plot(xfv, phifv, label='FV')
plt.plot(x_ex, phi_ex, label='Exact')
plt.legend(loc='best')
plt.show()