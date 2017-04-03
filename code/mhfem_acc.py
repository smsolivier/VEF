#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d 
from scipy.linalg import solve_banded 

import sys 

''' Mixed Hybrid Finite Element solver for moment equations (drift diffusion) ''' 

class MHFEM:

	def __init__(self, xe, Sigmaa, Sigmat, BCL=0, BCR=1, CENT=0):
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

		self.N = np.shape(xe)[0] # number of cell edges 

		self.n = 2*self.N - 1 # number of rows and columns of A 

		self.xc = np.zeros(self.N-1) # store cell centers 
		self.xe = xe # cell edges 

		# get cell centers 
		for i in range(self.N-1):

			# midpoint between cell edges 
			self.xc[i] = (self.xe[i] + self.xe[i+1])/2 

		# combine edge and center points 
		self.x = np.sort(np.concatenate((self.xc, self.xe)))

		# material properties 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 

		# boundary conditions 
		self.BCL = BCL 
		self.BCR = BCR 

		self.CENT = CENT 

		# create banded coefficient matrix, bandwidth 5 
		# upper diagonals have leading zeros, lower have trailing zeros 
		# A[0,:] = 2nd upper 
		# A[1,:] = 1st upper 
		# A[2,:] = diagonal 
		# A[3,:] = 1st lower 
		# A[4,:] = 2nd lower 
		self.A = np.zeros((5, self.n)) 

	def discretize(self, mu2, B):
		''' setup coefficient matrix with MHFEM equations ''' 

		# make <mu^2> function, cubic spline interpolation 
		if (self.CENT == 1):

			mu2f = interp1d(self.xc, mu2, fill_value='extrapolate') 

		elif (self.CENT == 0):

			mu2f = interp1d(self.xe, mu2)

		elif (self.CENT == 2):

			mu2f = interp1d(self.xc, mu2, fill_value='extrapolate')

		self.mu2f = mu2f 

		# build equations 
		for i in range(1, self.n, 2):

			hi = self.x[i+1] - self.x[i-1] # cell width, x_i+1/2 - x_i-1/2 
			beta = 2/(self.Sigmat(self.x[i])*hi)

			# balance equation 
			# lower diagonal 
			self.A[3,i-1] = -3 * beta * mu2f(self.x[i-1])

			# diagonal term 
			self.A[2,i] = 6*beta*mu2f(self.x[i]) + self.Sigmaa(self.x[i])*hi 

			# upper diagonal 
			self.A[1,i+1] = -3*beta*mu2f(self.x[i+1])

			# phi_i+1/2 equation
			if (i != self.n-2):

				# cell i+1 width, x_i+3/2 - x_i+1/2 
				h1 = self.x[i+3] - self.x[i+1] 
				beta1 = 2/(self.Sigmat(self.x[i+2])*h1) 

				# second lower (phi_i-1/2)
				self.A[4,i-1] = -beta*mu2f(self.x[i-1])

				# first lower (phi_i) 
				self.A[3,i] = 3*beta*mu2f(self.x[i])
				
				# diagonal term (phi_i+1/2)
				self.A[2,i+1] = -2*(beta + beta1)*mu2f(self.x[i+1])

				# first upper (phi_i+1)
				self.A[1,i+2] = 3*beta1*mu2f(self.x[i+2])

				# second upper (phi_i+3/2)
				self.A[0,i+3] = -beta1*mu2f(self.x[i+3])

		# boundary conditions 
		# left 
		if (self.BCL == 0): # reflecting 

			h1 = self.x[2] - self.x[0] 
			beta1 = 2/(self.Sigmat(self.x[1])*h1) 

			# J_1L = 0 
			# diagonal (phi_1/2)
			self.A[2,0] = -2*beta1*mu2f(self.x[0])

			# first upper (phi_1)
			self.A[1,1] = 3*beta1*mu2f(self.x[1])

			# second upper (phi_3/2)
			self.A[0,2] = -beta1*mu2f(self.x[2])

		elif (self.BCL == 1): # marshak 

			h1 = self.x[2] - self.x[0] 
			beta1 = 2/(self.Sigmat(self.x[1])*h1) 

			# diagonal (phi_1/2)
			self.A[2,0] = -B[0] - 2*beta1*mu2f(self.x[0]) 

			# first upper (phi_1)
			self.A[1,1] = 3*beta1*mu2f(self.x[1])

			# second upper (phi_3/2)
			self.A[0,2] = -beta1*mu2f(self.x[2])

		else:
			print('\n--- FATAL ERROR: MHFEM left boundary condition not defined ---\n')
			sys.exit()

		# right
		if (self.BCR == 0): # reflecting 

			hN = self.x[-1] - self.x[-3] # cell N width 
			betaN = 2/(self.Sigmat(self.x[-2])*hN)

			# J_NR = 0 
			# second lower (phi_N-1/2)
			self.A[4,-3] = betaN*mu2f(self.x[-3])

			# first lower (phi_N)
			self.A[3,-2] = -3*betaN*mu2f(self.x[-2])

			# diagonal (phi_N+1/2)
			self.A[2,-1] = 2*betaN*mu2f(self.x[-1])

		elif (self.BCR == 1): # marshak 

			hN = self.x[-1] - self.x[-3] # cell N width 
			betaN = 2/(self.Sigmat(self.x[-2])*hN)

			# second lower (phi_N-1/2)
			self.A[4,-3] = betaN*mu2f(self.x[-3])

			# first lower (phi_N)
			self.A[3,-2] = -3*betaN*mu2f(self.x[-2])

			# diagonal (phi_N+1/2)
			self.A[2,-1] = B[-1] + 2*betaN*mu2f(self.x[-1]) 

		else:
			print('\n --- FATAL ERROR: MHFEM right boundary condition not defined ---\n')
			sys.exit()

	def discretizeGauss(self, mu2, B):
		''' setup coefficient matrix with MHFEM equations 
			Inputs:
				mu2: array of cell edge and center Eddington factors 
				B: boundary value (set to 1/2 for marshak)
		''' 

		# build equations 
		for i in range(1, self.n, 2):

			hi = self.x[i+1] - self.x[i-1] # cell width, x_i+1/2 - x_i-1/2 
			beta = 2/(self.Sigmat(self.x[i])*hi)

			# balance equation 
			# lower diagonal 
			self.A[3,i-1] = -3 * beta * mu2[i-1]

			# diagonal term 
			self.A[2,i] = 6*beta*mu2[i] + self.Sigmaa(self.x[i])*hi 

			# upper diagonal 
			self.A[1,i+1] = -3*beta*mu2[i+1]

			# phi_i+1/2 equation
			if (i != self.n-2):

				# cell i+1 width, x_i+3/2 - x_i+1/2 
				h1 = self.x[i+3] - self.x[i+1] 
				beta1 = 2/(self.Sigmat(self.x[i+2])*h1) 

				# second lower (phi_i-1/2)
				self.A[4,i-1] = -beta*mu2[i-1]

				# first lower (phi_i) 
				self.A[3,i] = 3*beta*mu2[i]
				
				# diagonal term (phi_i+1/2)
				self.A[2,i+1] = -2*(beta + beta1)*mu2[i+1]

				# first upper (phi_i+1)
				self.A[1,i+2] = 3*beta1*mu2[i+2]

				# second upper (phi_i+3/2)
				self.A[0,i+3] = -beta1*mu2[i+3]

		# boundary conditions 
		# left 
		if (self.BCL == 0): # reflecting 

			h1 = self.x[2] - self.x[0] 
			beta1 = 2/(self.Sigmat(self.x[1])*h1) 

			# J_1L = 0 
			# diagonal (phi_1/2)
			self.A[2,0] = -2*beta1*mu2[0] 

			# first upper (phi_1)
			self.A[1,1] = 3*beta1*mu2[1]

			# second upper (phi_3/2)
			self.A[0,2] = -beta1*mu2[2]

		elif (self.BCL == 1): # marshak 

			h1 = self.x[2] - self.x[0] 
			beta1 = 2/(self.Sigmat(self.x[1])*h1) 

			# diagonal (phi_1/2)
			self.A[2,0] = -B[0] - 2*beta1*mu2[0]

			# first upper (phi_1)
			self.A[1,1] = 3*beta1*mu2[1]

			# second upper (phi_3/2)
			self.A[0,2] = -beta1*mu2[2]

		else:
			print('\n--- FATAL ERROR: MHFEM left boundary condition not defined ---\n')
			sys.exit()

		# right
		if (self.BCR == 0): # reflecting 

			hN = self.x[-1] - self.x[-3] # cell N width 
			betaN = 2/(self.Sigmat(self.x[-2])*hN)

			# J_NR = 0 
			# second lower (phi_N-1/2)
			self.A[4,-3] = betaN*mu2[-3]

			# first lower (phi_N)
			self.A[3,-2] = -3*betaN*mu2[-2]

			# diagonal (phi_N+1/2)
			self.A[2,-1] = 2*betaN*mu2[-1]

		elif (self.BCR == 1): # marshak 

			hN = self.x[-1] - self.x[-3] # cell N width 
			betaN = 2/(self.Sigmat(self.x[-2])*hN)

			# second lower (phi_N-1/2)
			self.A[4,-3] = betaN*mu2[-3]

			# first lower (phi_N)
			self.A[3,-2] = -3*betaN*mu2[-2]

			# diagonal (phi_N+1/2)
			self.A[2,-1] = B[-1] + 2*betaN*mu2[-1]

		else:
			print('\n --- FATAL ERROR: MHFEM right boundary condition not defined ---\n')
			sys.exit()

	def getEdges(self, phi):
		''' Convert a combined edge and center array to edges only ''' 

		# get edge values 
		phiEdge = np.zeros(self.N)

		ii = 0 
		for i in range(0, self.n, 2):

			phiEdge[ii] = phi[i]

			ii += 1 

		return phiEdge

	def getCenters(self, phi):
		''' Convert a combined edge and center array to centers only ''' 

		# get center values 
		phiCent = np.zeros(self.N-1) 

		ii = 0
		for i in range(1, self.n, 2):

			phiCent[ii] = phi[i]

			ii += 1 

		return phiCent

	def solve(self, q, qq):
		''' Compute phi = A^-1 q with banded solver 
			Inputs:
				q: cell centered array of source terms 
				qq: cell centered array of first moment of source 
		''' 

		# check q and qq are cell centered 
		if (np.shape(q)[0] != self.N-1):

			print('\n--- FATAL ERROR: MHFEM q must be cell centered ---\n')
			sys.exit()

		if (np.shape(qq)[0] != self.N-1):

			print('\n--- FATAL ERROR: MHFEM qq must be cell centered ---\n') 
			sys.exit()

		ii = 0 # store iterations of q 
		b = np.zeros(self.n) # store source vector 
		# set odd equations to the source, leave even as zero 
		for i in range(1, self.n, 2):

			b[i] = q[ii] * (self.x[i+1] - self.x[i-1])

			ii += 1 

		# set even equations to use first moment of q 
		ii = 0 
		for i in range(1, self.n-2, 2):

			beta = 2/(self.Sigmat(self.x[i]))
			beta1 = 2/(self.Sigmat(self.x[i+2])) 

			b[i+1] = .5*(beta1*qq[ii+1] - beta*qq[ii])

			ii += 1 

		# set boundary b 
		beta1 = 2/(self.Sigmat(self.x[1]))
		b[0] = beta1/2*qq[0] 

		betaN = 2/(self.Sigmat(self.x[-2]))
		b[-1] = betaN/2*qq[-1]

		# solve for flux 
		# solve banded matrix 
		phi = solve_banded((2,2), self.A, b)

		# check solution 
		# self.checkSolution(phi, self.mu2f(self.x), q)

		if (self.CENT == 0): # return edges only 

			return self.xe, self.getEdges(phi)

		elif (self.CENT == 1): # return centers only 

			return self.xc, self.getCenters(phi)

		else: # return edges and centers 

			return self.x, phi 

	def checkSolution(self, phi, mu2, q):
		''' Check for continuity and conservation 
			Only supports isotropic q (qq = 0) 
		''' 

		# continuity 
		Jl = np.zeros(self.N - 1) 
		Jr = np.zeros(self.N - 1)

		phiEdge = self.getEdges(phi)
		muEdge = self.getEdges(mu2)

		Fedge = phiEdge * muEdge

		phiCent = self.getCenters(phi)
		muCent = self.getCenters(mu2) 

		Fcent = phiCent * muCent 

		for i in range(self.N-1):

			h = self.xe[i+1] - self.xe[i]

			Jr[i] = -2/(self.Sigmat(self.xc[i])*h) * (
				Fedge[i] - 3*Fcent[i] + 2*Fedge[i+1])

			Jl[i] = -2/(self.Sigmat(self.xc[i])*h) * (
				-2*Fedge[i] + 3*Fcent[i] - Fedge[i+1])

		cont = np.zeros(self.N - 2)

		for i in range(1, self.N - 1):

			cont[i-1] = np.fabs((Jl[i] - Jr[i-1])/Jr[i-1])

		# conservation 
		balance = np.zeros(self.N-1)
		for i in range(self.N-1):

			h = self.xe[i+1] - self.xe[i] 

			qq = .5*(q[i] + q[i+1]) * h

			balance[i] = np.fabs(Jr[i] - Jl[i] + 
				self.Sigmaa(self.xc[i])*phiCent[i]*h - qq)

			balance[i] /= qq

		tol = 1e-3

		if (np.max(cont) > tol):

			print('--- WARNING: MHFEM continuity of current broken ---')

		if (np.max(balance) > tol):

			print('--- WARNING: MHFEM conservation broken ---')


if __name__ == '__main__':

	from exactDiff import * 

	xb = 10

	eps = 1
	Sigmaa = .1*eps 
	Sigmat = 1
	Sigmatf = lambda x: .5*(x < xb/2) + 1*(x >= xb/2) 

	Q = 1 * eps 

	BCL = 1
	BCR = 1 

	N = 25 # number of edges 
	xe = np.linspace(-xb, xb, N)
	mu2 = np.ones(N-1)/3 
	mhfem = MHFEM(xe, lambda x: Sigmaa, Sigmatf, BCL, BCR, CENT=2)
	mhfem.discretize(mu2, np.ones(N)/2)
	x, phi = mhfem.solve(np.ones(N-1)*Q, np.zeros(N-1))

	phiEdge = mhfem.getEdges(phi)
	phiCent = mhfem.getCenters(phi)

	xEdge = mhfem.getEdges(x)
	xCent = mhfem.getCenters(x)

	phiAvg = np.zeros(len(phiCent))

	for i in range(1, len(phiEdge)):

		phiAvg[i-1] = .5*(phiEdge[i] + phiEdge[i-1])

	phi_ex = exactDiff(Sigmaa, Sigmat, Q, xb, BCL, BCR)

	print(np.linalg.norm(phiEdge - phi_ex(xEdge), 2))
	print(np.linalg.norm(phiCent - phi_ex(xCent), 2))

	plt.subplot(1,2,1)
	plt.plot(x, phi, '-o', label='MHFEM')
	plt.plot(x, phi_ex(x), label='Exact')
	plt.xlabel('x')
	plt.ylabel(r'$\phi$')
	plt.legend(loc='best')

	plt.subplot(1,2,2)
	plt.semilogy(xEdge, np.fabs(phiEdge - phi_ex(xEdge))/phi_ex(xEdge), '-o', label='Edge')
	plt.semilogy(xCent, np.fabs(phiCent - phi_ex(xCent))/phi_ex(xCent), '-o', label='Center')
	# plt.semilogy(x, np.fabs(phi - phi_ex(x)), '-o')
	plt.xlabel('x')
	plt.ylabel('| MHFEM - Exact | / Exact ')
	plt.legend(loc='best')
	plt.show()