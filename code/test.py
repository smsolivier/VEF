#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_acc import * 

N = 5
xb = 1 
mh = MHFEM(np.linspace(0, xb, N), np.ones(N)/3, lambda x: .1, lambda x: .83, BCL=1, BCR=1)

x, phi = mh.solve(np.ones(N))

print((mh.A.transpose() == mh.A).all())

np.savetxt('a.txt', mh.A, delimiter=',')

plt.imshow(mh.A, interpolation='none', cmap='viridis')
plt.colorbar()
plt.show()