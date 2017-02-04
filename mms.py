#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_diff import *
from fv_diff import *

Sigmaa = .1 
Sigmat = .83 

D = 1/(3*Sigmat)

xb = 1 

N = np.array([80, 160])

L = 2*xb
phi_ex = lambda x: np.cos(np.pi*x/L)
Qmms = lambda x: D*(np.pi/L)**2 * phi_ex(x) + Sigmaa*phi_ex(x) 

def getOrder(solver):

	err_phi = np.zeros(len(solver))
	for i in range(len(solver)):

		x, phi = solver[i].solve(Qmms(solver[i].x))

		err_phi[i] = np.linalg.norm(phi - phi_ex(x), 2)

	order = np.log(err_phi[0]/err_phi[1])/np.log(2)

	print('order =', order)

	return err_phi, order

getOrder([MHFEM(x, BCL=0, BCR=0) for x in N])

getOrder([finiteVolume(x, lambda x: Sigmat, lambda x: Sigmaa) for x in N])

# mhfem = MHFEM(50, BCL=0, BCR=0)
# x, phi = mhfem.solve(Qmms(mhfem.x))
# x_ex = np.linspace(0, xb, 100)

# plt.plot(x_ex, phi_ex(x_ex), '--')
# plt.plot(x, phi)
# plt.show()