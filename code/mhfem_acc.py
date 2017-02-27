#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d 
from scipy.linalg import solve_banded 

import sys 

class MHFEM:

	def __init__(self, xe, mu2, Sigmaa, Sigmat, B, BCL=0, BCR=1):
		''' solves moment equations with general <mu^2> 
			Inputs:
				xe: array of cell edges 
				mu2: array of <mu^2> values at cell edges (will be interpolated)
				Sigmaa: absorption XS function 
				Sigmat: total XS function 
				B: array of boundary Eddington values for transport consistency 
					set to 1/2 for marshak 
				BCL: left boundary 
					0: reflecting 
					1: marshak 
				BCR: right boundary 
					0: reflecting 
					1: marshak 
		''' 

		self.B = B # transport consistent boundary 

		N = np.shape(xe)[0] # number of cell edges 

		n = 2*N - 1 # number of rows and columns of A 

		xc = np.zeros(N-1) # store cell centers 
		for i in range(1, N):
			xc[i-1] = (xe[i] + xe[i-1])/2 # midpoint between cell edges 

		# combine edge and center points 
		x = np.sort(np.concatenate((xc, xe)))

		# make <mu^2> function 
		mu2f = interp1d(xe, mu2) 

		# store banded matrix entries  
		# upper diagonals have leading zeros, lower have trailing zeros 
		# A[0,:] = 2nd upper 
		# A[1,:] = 1st upper 
		# A[2,:] = diagonal 
		# A[3,:] = 1st lower 
		# A[4,:] = 2nd lower 
		A = np.zeros((5, n)) # coefficient matrix 

		# build equations 
		for i in range(1, n, 2):

			hi = x[i+1] - x[i-1] # cell width, x_i+1/2 - x_i-1/2 

			# balance equation 
			# lower diagonal 
			A[3,i-1] = -6/(Sigmat(x[i])*hi)*mu2f(x[i-1])

			# diagonal term 
			A[2,i] = Sigmaa(x[i])*hi + 12/(Sigmat(x[i])*hi)*mu2f(x[i])

			# upper diagonal 
			A[1,i+1] = -6/(Sigmat(x[i])*hi)*mu2f(x[i+1]) 

			# phi_i+1/2 equation
			if (i != n-2):

				h1 = x[i+3] - x[i+1] # cell i+1 width, x_i+3/2 - x_i+1/2 

				# second lower (phi_i-1/2)
				A[4,i-1] = -2/(Sigmat(x[i])*hi)*mu2f(x[i-1])

				# first lower (phi_i) 
				A[3,i] = 6/(Sigmat(x[i])*hi)*mu2f(x[i])
				
				# diagonal term (phi_i+1/2)
				A[2,i+1] = -4*(1/(Sigmat(x[i])*hi) + 1/(Sigmat(x[i+2])*h1))*mu2f(x[i+1])

				# first upper (phi_i+1)
				A[1,i+2] = 6/(Sigmat(x[i+2])*h1)*mu2f(x[i+2])

				# second upper (phi_i+3/2)
				A[0,i+3] = -2/(Sigmat(x[i+2])*h1)*mu2f(x[i+3])

		# boundary conditions 
		# left 
		if (BCL == 0): # reflecting 

			# J_1L = 0 
			# diagonal (phi_1/2)
			A[2,0] = -2*mu2f(x[0])

			# first upper (phi_1)
			A[1,1] = 3*mu2f(x[1])

			# second upper (phi_3/2)
			A[0,2] = -1*mu2f(x[2])

		elif (BCL == 1): # marshak 

			alpha = 4/(Sigmat(x[1])*(x[2] - x[0])) 

			# diagonal (phi_1/2)
			A[2,0] = self.B[0] + alpha*mu2f(x[0])

			# first upper (phi_1)
			A[1,1] = -3/2*alpha*mu2f(x[1])

			# second upper (phi_3/2)
			A[0,2] = .5*alpha*mu2f(x[2])

		else:
			print('left boundary condition not defined')
			sys.exit()

		# right
		if (BCR == 0): # reflecting 

			# J_NR = 0 
			# second lower (phi_N-1/2)
			A[4,-3] = mu2f(x[-3])

			# first lower (phi_N)
			A[3,-2] = -3*mu2f(x[-2])

			# diagonal (phi_N+1/2)
			A[2,-1] = 2*mu2f(x[-1])

		elif (BCR == 1): # marshak 

			alpha = 4/(Sigmat(x[-2])*(x[-1] - x[-3]))

			# second lower (phi_N-1/2)
			A[4,-3] = .5*alpha*mu2f(x[-3])

			# first lower (phi_N)
			A[3,-2] = -3/2*alpha*mu2f(x[-2])

			# diagonal (phi_N+1/2)
			A[2,-1] = self.B[-1] + alpha*mu2f(x[-1])

		else:
			print('right boundary condition not defined')
			sys.exit()

		# make variables public 
		self.A = A 
		self.n = n 
		self.N = N 
		self.x = x 
		self.xe = xe 

	def solve(self, q):

		ii = 0 # store iterations of q 
		b = np.zeros(self.n) # store source vector 
		# set odd equations to the source, leave even as zero 
		for i in range(1, self.n-1, 2):

			b[i] = (q[ii] + q[ii+1])/2 * (self.x[i+1] - self.x[i-1])

			ii += 1 

		# solve for flux 
		# solve banded matrix 
		phi = solve_banded((2,2), self.A, b)

		# get edge values 
		phiEdge = np.zeros(self.N)

		ii = 0 
		for i in range(0, self.n, 2):

			phiEdge[ii] = phi[i]

			ii += 1 

		return self.xe, phiEdge

if __name__ == '__main__':

	Sigmaa = .1 
	Sigmat = .83 

	xb = 1

	Q = 1 

	N = 10 # number of cells 
	xe = np.linspace(0, xb, N+1)
	mu2 = np.ones(N+1)/3 
	mhfem = MHFEM(xe, mu2, lambda x: Sigmaa, lambda x: Sigmat, BCL=0, BCR=1)
	x, phi = mhfem.solve(np.ones(N)*Q)

	# exact solution 
	D = 1/(3*Sigmat) 
	L = np.sqrt(D/Sigmaa)
	c1 = -Q/Sigmaa/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	phi_ex = lambda x: c1*np.cosh(x/L) + Q/Sigmaa 
	x_ex = np.linspace(0, xb, 100)

	plt.plot(x_ex, phi_ex(x_ex), '--')
	plt.plot(x, phi)
	# plt.plot(x, np.fabs(phi - phi_ex(x)), '-o')
	# plt.yscale('log')
	plt.show()

	# check order of convergence 
	N = np.array([20, 40, 80, 160])
	mh = [MHFEM(np.linspace(0, xb, x), np.ones(x)/3, 
		lambda x: Sigmaa, lambda x: Sigmat) for x in N]

	err = np.zeros(len(N))
	for i in range(len(N)):

		x, phi = mh[i].solve(np.ones(N[i])*Q)

		err[i] = np.linalg.norm(phi - phi_ex(x), 2)

		plt.plot(x, np.fabs(phi - phi_ex(x)))

	plt.yscale('log')
	plt.show()

	fit = np.polyfit(np.log(1/(N-1)), np.log(err), 1)

	print(fit[0])

	plt.loglog(1/(N-1), np.exp(fit[1]) * (1/(N-1))**fit[0], '-o')
	plt.loglog(1/(N-1), err, '-o')
	plt.show()




