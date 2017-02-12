#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from TDMA import * 

import sys 

# import solver classes 
from mhfem_diff import * 
from fv_diff import * 
from fem import * 

def powerIteration(solver, nuSigmaf=.125, tol=1e-8, LOUD=False):
	''' use power iteration to solve for fundamental mode flux and k ''' 

	# initial guess 
	phi = np.ones(solver.N) # array of 1's 

	h = solver.h 

	# normalize
	phi /= np.sum(phi*h) # make norm(phi) = 1 

	kold = 0 
	ii = 0
	while(True):

		# solve diffusion eq with solver class's solver 
		x, phi_new = solver.solve(nuSigmaf*phi) # h multplied in solver function 

		# compute k with new phi 
		k = np.sum(phi_new*h)

		# normalize phi_new 
		phi_new /= k 

		# test convergence 
		phiConv = np.linalg.norm(phi_new - phi, 2)/np.linalg.norm(phi_new, 2)
		kConv = np.fabs(k - kold)/k 
		if (phiConv	< tol and kConv < tol):

			break # exit loop if converged 

		# update 
		phi = np.copy(phi_new) # update old phi 
		kold = k # update old k 

		# print iterations 
		if (LOUD):
			fmt = '{:.6}'
			print(fmt.format(phiConv) + '\t' + fmt.format(kConv) + '\t' + fmt.format(k))

		ii += 1 # count iterations 

	return x, phi/np.max(phi), k

def getOrder(solver, Sigmaa, Sigmat, nuSigmaf):
	''' find the order of convergence of solver 
		solver is a list of solver objects with varying number of cells 
	''' 

	err_k = np.zeros(len(solver)) # store k errors 
	err_phi = np.zeros(len(solver)) # store phi errors 

	# exact solution parameters 
	D = 1/(3*Sigmat) 
	k_ex = 4*xb**2*nuSigmaf/(D*np.pi**2 + 4*xb**2*Sigmaa)
	a = np.sqrt((nuSigmaf - k_ex*Sigmaa)/(k_ex*D))

	N = np.array([x.N for x in solver])

	# solve and compare to exact for each N 
	for i in range(len(solver)): # loop through N 

		# solve at each N 
		x, phi, k = powerIteration(solver[i], nuSigmaf)

		# exact solution, evaluate at same x's 
		phi_ex = np.cos(a*x)

		# k error 
		err_k[i] = np.fabs(k_ex - k)/k_ex 

		# phi error 
		err_phi[i] = np.sqrt(np.sum((phi_ex - phi)**2)/np.sum(phi_ex**2))

	# get slope of best fit line 
	fit = np.polyfit(np.log(1/N), np.log(err_phi), 1)
	fitk = np.polyfit(np.log(1/N), np.log(err_k), 1)

	return err_phi, fit[0], fitk[0]

# material properties 
Sigmaa = .1 
Sigmat = .83 
nuSigmaf = .125 

# properties as functions 
Sigmaa_f = lambda x: Sigmaa 
Sigmat_f = lambda x: Sigmat 

# length of domain 
xb = 1 

# boundary conditions 
BCL = 0 # reflecting on left 
BCR = 1 # zero flux on right 

# test order of convergence  
N = np.array([40, 80, 160, 320]) # number of cells to use 

# list of solver objects 
mhfem = [MHFEM(x, Sigmaa, Sigmat, xb=xb, BCL=BCL, BCR=BCR) for x in N] 
fv = [finiteVolume(x, Sigmaa_f, Sigmat_f, xb=xb, BCL=BCL, BCR=BCR) for x in N] 
# fe = [FEM(np.linspace(0, xb, x), np.ones(x)/3, lambda x: Sigmaa, lambda x: Sigmat) for x in N]

# get error and order for fv and mhfem 
fv_err, fv_ord, fv_ordk = getOrder(fv, Sigmaa, Sigmat, nuSigmaf)
fe_err, fe_ord, fe_ordk = getOrder(mhfem, Sigmaa, Sigmat, nuSigmaf)
# fe2_err, fe2_ord, fe2_ordk = getOrder(fe, Sigmaa, Sigmat, nuSigmaf) 

print('FV order =', fv_ord, fv_ordk)
print('MHFEM order =', fe_ord, fe_ordk)

# plot error v cell size 
# plt.loglog(1/N, fv_err, '-o', label='FV')
# plt.loglog(1/N, fe_err, '--o', label='MHFEM')
# plt.legend(loc='best')
# plt.show()

# --- analytic solution ---
D = 1/(3*Sigmat) 
L = xb + 2*D 
# k_ex = 4*L**2*nuSigmaf/(D*np.pi**2 + 4*L**2*Sigmaa)
# a = np.sqrt((nuSigmaf - k_ex*Sigmaa)/(k_ex*D))
# phi_ex = lambda x: np.cos(a*x)

x_ex = np.linspace(0, xb, 100)
atilde = xb
B2 = (np.pi/(2*atilde))**2 
k_ex = nuSigmaf/(D*B2 + Sigmaa)
phi_ex = lambda x: np.cos(np.pi/(2*atilde)*x)

# plot flux shape 
N = 50
# finite volume solve 
x, phi, kfv = powerIteration(fv[-1], nuSigmaf)
plt.plot(x, phi, label='FV')
# fhfem solve 
x, phi, kfem = powerIteration(mhfem[-1], nuSigmaf)
print('k_fv =', kfv)
print('k_fe =', kfem)
print('k_ex =', k_ex)
plt.plot(x, phi, label='MHFEM')
plt.plot(x_ex, phi_ex(x_ex), '--', label='exact')
plt.legend(loc='best')
plt.show()