#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

from TDMA import * 

class finiteVolume:

	def __init__(self, N, Sigmaa, Sigmat, xb=1, BCL=0, BCR=0, LOUD=False):
		''' initialize finite volume spatial discretization ''' 

		h = xb/N # grid spacing 

		x = np.linspace(h/2, xb - h/2, N) # cell centroid locations 

		# i-1 cell coefficient 
		aW = np.zeros(N)
		for i in range(1, N):
			aW[i] = 2/(3*h*(Sigmat(x[i]) + Sigmat(x[i-1])))

		# i+1 coefficient
		aE = np.zeros(N)
		for i in range(N-1):
			aE[i] = 2/(3*h*(Sigmat(x[i+1]) + Sigmat(x[i])))

		Sp = Sigmaa(x) * h # phi dependent source 

		# boundary conditions 
		ab = np.zeros(N) # store boundary conditions 

		# left boundary 
		D0 = 1/(3*Sigmat(x[0])) # D at left boundary 
		if (BCL == 0): # reflecting 
			ab[0] = 0 
		elif (BCL == 1): # zero flux 
			ab[0] = 2*D0/h 
		elif (BCL == 2): # marshak 
			ab[0] = 2*D0/(4*D0 + 4*h)
		else:
			print('left boundary not defined')
			sys.exit()

		# right boundary 
		DN = 1/(3*Sigmat(x[-1])) # D at right boundary 
		if (BCR == 0): # reflecting BC 
			ab[-1] = 0
		elif (BCR == 1): # zero flux BC
			ab[-1] = 2*DN/h 
		elif (BCR == 2): # marshak condition 
			ab[-1] = 2*DN/(4*DN + 4*h)
		else:
			print('right boundary not defined')
			sys.exit()

		# coefficient for cell i 
		ap = aW + aE + ab + Sp 

		# make public 
		self.N = N 
		self.h = h 
		self.x = x
		self.aW = aW 
		self.ap = ap
		self.aE = aE 

	def solve(self, b):
		''' solve for flux with tri diagonal solver ''' 

		phi_new = TDMA(-self.aW, self.ap, -self.aE, b*self.h)

		return self.x, phi_new 

if __name__ == '__main__':

	Sigmat = .83 
	Sigmaa = .1 

	xb = 1 

	Q = 1 

	N = 500 # number of volumes 
	fv = finiteVolume(N, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2) 

	x, phi = fv.solve(np.ones(N)*Q) # solve for flux with uniform Q 

	# exact solution 
	D = 1/(3*Sigmat)
	L = np.sqrt(D/Sigmaa)
	atilde = xb + 2*D 

	phi_ex = lambda x: Q/Sigmaa*(1 - np.cosh(x/L)/np.cosh(atilde/L))

	# plot 
	plt.plot(x, phi_ex(x), '--', label='Exact')
	plt.plot(x, phi, label='FV')
	plt.legend(loc='best')
	plt.show()