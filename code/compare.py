#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_diff import * 
from fv_diff import * 
from fem import * 

''' compare MHFEM to finite volume ''' 

N = 1000

xb = 50

Sigmaa = .1 
Sigmat = .83 
q = .1 

bc = 1

# initialize finite volume with marshak condition 
fv = finiteVolume(N, lambda x: Sigmaa, lambda x: Sigmat, xb, BCL=0, BCR=2)

# initialize MHFEM with marshak condition 
fe = MHFEM(N, Sigmaa, Sigmat, xb, BCL=0, BCR=2)

# initialize regular fem 
fe2 = FEM(N, Sigmaa, Sigmat, xb)

# solve for flux 
b = np.ones(N)*q 
xfv, phifv = fv.solve(b)
xfe, phife = fe.solve(b)
xfe2, phife2 = fe2.solve(b)

plt.plot(xfv, phifv, label='FV')
plt.plot(xfe, phife, label='MHFEM')
plt.plot(xfe2, phife2, label='FEM')
plt.legend(loc='best')
plt.show()