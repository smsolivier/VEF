#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

class MHFEM:

	def __init__(self, N, sigma_a=.1, sigma_t=.83, xb=1, BCL=0, BCR=0):
		''' initialize MHFEM spatial discretization 
			BC:
				0: reflecting
				1: zero flux 
				2: marshak 
		''' 

		self.N = N

		self.n = 1 + 2*N # number of elements per row of coefficient matrix 

		self.h = xb/N # cell spacing, uniform 
		# self.x = np.linspace(0, xb, self.n) # array of n evenly spaced points 
		self.x = np.linspace(self.h/2, xb - self.h/2, N)

		# build A matrix 
		A = np.zeros((self.n, self.n)) # coefficient matrix  

		for i in range(1, self.n, 2):

			# set phi_i equation
				A[i,i-1:i+2] = np.array([
					-2/(sigma_t*self.h), # phi_i-1/2 
					4/(sigma_t*self.h) + sigma_a*self.h, # phi_i 
					-2/(sigma_t*self.h)
					]) # phi_i+1/2 

				# set phi_i+1/2 equation
				if (i != self.n-2): # don't overwrite phi_N+1/2 equation 
					A[i+1,i-1:i+4] = np.array([
						1, # phi_i-1/2 
						-3, # phi_i 
						4, # phi_i+1/2 
						-3, # phi_i+1 
						1 # phi_i+3/2 
						])

		# boundary conditions 
		# left 
		if (BCL == 0): # reflecting 
			A[0,:3] = np.array([-2, 3, -1]) # set J_1L = 0 

		elif (BCL == 1): # zero flux 
			A[0,0] = 1 # phi_1/2 = 0 

		elif (BCL == 2): # marshak 
			alpha = -4/(3*sigma_t*self.h)
			# A[0,:2] = np.array([1 + 4/(3*sigma_t*self.h), -4/(3*sigma_t*self.h)])
			A[0,:3] = np.array([1 - 2*alpha, 3*alpha, -alpha])

		else:
			print('left boundary condition not defined')
			sys.exit()

		# right 
		if (BCR == 0): # reflecting 
			A[-1,-3:] = np.array([1, -3, 2]) # set J_NR = 0 

		elif (BCR == 1): # zero flux 
			A[-1,-1] = 1 # phi_N+1/2 = 0 

		elif (BCR == 2): # marshak 
			# -2D/h phi_N + (1 + 2D/h) phi_N+1/2 = 0 
			# A[-1, -2:] = np.array([-4/(3*sigma_t*self.h), 1 + 4/(3*sigma_t*self.h)])
			alpha = 4/(3*sigma_t*self.h)
			A[-1,-3:] = np.array([alpha, -3*alpha, 1 + 2*alpha])

		else:
			print('right boundary condition not defined')
			sys.exit()

		# make A public 
		self.A = A 

	def solve(self, q):
		''' solve A phi = b by inverting A 
			returns cell centered flux and spatial locations 
		''' 

		ii = 0 # store iterations of b 
		b = np.zeros(self.n) # store source vector 
		# set odd equations to the source, leave even as zero 
		for i in range(1, self.n-1, 2):

			b[i] = q[ii] * self.h

			ii += 1 

		assert(ii == self.N)

		# solve for new flux 
		phi = np.dot(np.linalg.inv(self.A), b)

		# extract cell center flux values 
		phiCenter = np.zeros(self.N)

		ii = 0 # store iterations of phiCenter 
		for i in range(1, self.n, 2):

			phiCenter[ii] = phi[i] 

			ii += 1

		assert(ii == self.N)

		return self.x, phiCenter

if __name__ == '__main__':

	Sigmaa = .1 
	Sigmat = .83 

	xb = 1 

	Q = 1 

	N = 100 # number of volumes 
	mhfem = MHFEM(N, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2) # initialize solver object 

	x, phi = mhfem.solve(np.ones(N)*Q) # solve for flux with uniform Q 

	# exact solution 
	D = 1/(3*Sigmat)
	L = np.sqrt(D/Sigmaa)
	atilde = 2*xb + 2*D 

	phi_ex = lambda x: Q/Sigmaa*(1 - np.cosh(x/L)/np.cosh(atilde/L))

	# plot 
	plt.plot(x, phi_ex(x), '--', label='Exact')
	plt.plot(x, phi, label='MHFEM')
	plt.legend(loc='best')
	plt.show()