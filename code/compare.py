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

N = np.array([40, 60, 80, 100])

# make solver objects 
mhfem = [MHFEM(np.linspace(0, xb, n), np.ones(n)/3, 
	lambda x: Sigmaa, lambda x: Sigmat) for n in N]
fem = [FEM(np.linspace(0, xb, n), np.ones(n)/3, 
	lambda x: Sigmaa, lambda x: Sigmat) for n in N]
fv = [finiteVolume(n, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2) for n in N]

def getOrder(sol, xeval):

	l2err = np.zeros(len(sol))
	err = np.zeros(len(sol))

	for i in range(len(sol)):

		x, phi = sol[i].solve(np.ones(N[i])*Q)

		l2err[i] = np.linalg.norm(phi - phi_ex(x), 2)
		# err[i] = np.sqrt(np.sum(phi - phi_ex(x))**2/np.sum(phi_ex(x)**2))
		
		f = interp1d(x, phi)

		err[i] = np.fabs(f(xeval) - phi_ex(xeval))/phi_ex(xeval)

	fit = np.polyfit(np.log(xb/N), np.log(err), 1)

	order = np.log(err[-2]/err[-1])/np.log(2)

	l2fit = np.polyfit(np.log(xb/N), np.log(l2err), 1)

	print(xeval, fit[0], l2fit[0], l2err[-1])

	return fit[0], l2fit[0]

# xeval = np.linspace(0, xb, 20)
# # xeval = np.copy(mhfem[0].xe)
# order = np.zeros(len(xeval))

# for i in range(len(xeval)):

# 	order[i], l2ord = getOrder(mhfem, xeval[i])

# plt.plot(xeval, order, '-o')
# plt.show()

xeval = xb/2
getOrder(mhfem, xeval)
getOrder(fem, xeval)
getOrder(fv, xeval)

# plt.show()
