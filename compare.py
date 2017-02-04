#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_diff import * 
from fv_diff import * 

N = 1000

xb = 1 

Sigmaa = .1 
Sigmat = .83 
q = .1 

bc = 1

# initialize finite volume with marshak condition 
fv = finiteVolume(N, lambda x: Sigmat, lambda x: Sigmaa, xb, BC=bc)

# initialize MHFEM with marshak condition 
fe = MHFEM(N, Sigmaa, Sigmat, xb, BC=bc)

# solve for flux 
b = np.ones(N)*q 
xfv, phifv = fv.solve(b)
xfe, phife = fe.solve(b)

plt.plot(xfv, phifv, label='FV')
plt.plot(xfe, phife, label='MHFEM')
plt.legend(loc='best')
plt.show()