#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

class MHFEM:

	def __init__(self, N, sigma_a, sigma_t, xb=1, BCL=0, BCR=0, EDGE=0):
		''' initialize MHFEM spatial discretization 
			BC:
				0: reflecting
				1: zero flux 
				2: marshak 
		''' 

		self.N = N
		self.EDGE = EDGE # return edge values or cell center values 
		self.xb = xb 

		self.n = 1 + 2*N # number of elements per row of coefficient matrix 

		self.h = xb/N 

		self.x = np.linspace(-self.h/2, xb-self.h/2, N)

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

		if (self.EDGE):
			# extract edges 
			phiEdge = np.zeros(self.N+1)

			ii = 0 
			for i in range(0, self.n, 2):

				phiEdge[ii] = phi[i] 

				ii += 1 

			return np.linspace(0, self.xb, self.N+1), phiEdge

		else:
			# extract cell center flux values 
			phiCenter = np.zeros(self.N)

			ii = 0 # store iterations of phiCenter 
			for i in range(1, self.n, 2):

				phiCenter[ii] = phi[i] 

				ii += 1

			return self.x, phiCenter

if __name__ == '__main__':

	Sigmaa = .1 
	Sigmat = .83 

	xb = 1 

	Q = 1 

	N = 10 # number of volumes 
	mhfem = MHFEM(N, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2, EDGE=0) # initialize solver object 

	x, phi = mhfem.solve(np.ones(N)*Q) # solve for flux with uniform Q 

	# exact solution 
	D = 1/(3*Sigmat) 
	L = np.sqrt(D/Sigmaa)
	c1 = -Q/Sigmaa/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	phi_ex = lambda x: c1*np.cosh(x/L) + Q/Sigmaa 

	# plot 
	# plt.plot(x, phi_ex(x), '--', label='Exact')
	# plt.plot(x, phi, label='MHFEM')
	# plt.legend(loc='best')
	# plt.show()

	# check order of convergence 
	N = np.array([400, 800, 1000])
	mh = [MHFEM(x, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2, EDGE=1) for x in N]

	err = np.zeros(len(N))
	for i in range(len(N)):

		x, phi = mh[i].solve(np.ones(N[i])*Q)

		err[i] = np.linalg.norm(phi - phi_ex(x), 2)

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0])

	plt.loglog(1/N, np.exp(fit[1]) * (1/N)**fit[0], '-o')
	plt.loglog(1/N, err, '-o')
	plt.show()