#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_acc import * 
from fv_diff import * 
from fem import * 

from scipy.interpolate import interp1d

''' compare MHFEM to finite volume ''' 

Sigmaa = .1 
Sigmat = .83 

xb = 1

Q = 1 

# exact solution 
D = 1/(3*Sigmat) 
L = np.sqrt(D/Sigmaa)
c1 = -Q/Sigmaa/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
phi_ex = lambda x: c1*np.cosh(x/L) + Q/Sigmaa 
x_ex = np.linspace(0, xb, 100)

N = np.array([50, 100, 200, 400])

# make solver objects 
mhfem = [MHFEM(np.linspace(0, xb, x), np.ones(x)/3, 
	lambda x: Sigmaa, lambda x: Sigmat) for x in N]
fem = [FEM(np.linspace(0, xb, x), np.ones(x)/3, lambda x: Sigmaa, lambda x: Sigmat) for x in N]
fv = [finiteVolume(x, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2) for x in N]

def getOrder(sol, xeval):

	err = np.zeros(len(sol))

	for i in range(len(sol)):

		x, phi = sol[i].solve(np.ones(N[i])*Q)

		# err[i] = np.linalg.norm(phi - phi_ex(x), 2)
		# err[i] = np.sqrt(np.sum(phi - phi_ex(x))**2/np.sum(phi_ex(x)**2))
		
		f = interp1d(x, phi)

		err[i] = np.fabs(f(xeval) - phi_ex(xeval))/phi_ex(xeval)

	fit = np.polyfit(np.log(xb/N), np.log(err), 1)

	order = np.log(err[-2]/err[-1])/np.log(2)

	print(fit[0], order)

	return fit[0]

getOrder(mhfem, xb/2)
getOrder(fem, xb/2)
getOrder(fv, xb/2)

# plt.show()
