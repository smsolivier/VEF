#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_diff import * 
from mc import * 

Sigmaa = .01 # absorption 
Sigmat = 1 # total 
Sigmas = Sigmat - Sigmaa # scattering 
q = 1 # source 

print('c =', Sigmas/Sigmat)

N = 250 # number of spatial points 
xb = 1 # width of domain 
x = np.linspace(0, xb, N) # x locations  
dx = xb/N # cell size 

def sweep(phi):

	psiL = np.zeros(N) # left moving angular flux 
	psiL[-1] = 0 # vacuum boundary condition (no left moving incident on left side) 

	# sweep left to right 
	for i in range(N-2, -1, -1):

		psiL[i] = (1 - .5*Sigmat*dx)*psiL[i+1] + \
			Sigmas/4*dx*(phi[i] + phi[i+1]) + q*dx/2 

		psiL[i] /= 1 + .5*Sigmat*dx 

	psiR = np.zeros(N) # right moving angular flux 
	psiR[0] = psiL[0] # set reflecting boundary condition 

	# sweep right to left 
	for i in range(1, N):

		psiR[i] = (1 - .5*Sigmat*dx)*psiR[i-1] + \
			Sigmas/4*dx*(phi[i] + phi[i-1]) + q*dx/2 

		psiR[i] /= 1 + .5*Sigmat*dx 

	# compute phi = int psi du 
	new_phi = psiR + psiL 

	return new_phi 

tol = 1e-8
phi = np.zeros(N) # initial guess for flux 
it = 0 # store number of iterations 
# loop until flux converges 
while (True):

	phi_new = sweep(phi) # determine flux using old flux 

	# check for convergence 
	if (np.linalg.norm(phi_new - phi, 2) < tol):
		break 

	# update old flux 
	phi = np.copy(phi_new) 

	it += 1 # increase number of iterations 

print('number of iterations =', it)

# exact solution 
x_ex = np.linspace(0, xb, 100)
alpha = np.sqrt(Sigmaa*Sigmat)
a = -Sigmat*q/(Sigmaa*(Sigmat*np.cosh(alpha*xb) + alpha*np.sinh(alpha*xb)))
phi_ex = a*np.cosh(alpha*x_ex) + q/Sigmaa 

# compute monte carlo solution 
xmc, flux, leakL, leakR = montecarlo(1000, 20, 50, Sigmat, Sigmas, Sigmaa, q, xb=xb)

plt.plot(x, phi, label=r'S$_2$')
plt.errorbar(xmc, flux[:,0], yerr=flux[:,1], label='MC')
plt.plot(x_ex, phi_ex, label='Exact')
plt.xlabel('x')
plt.ylabel('Flux')
plt.legend(loc='best')
plt.show()