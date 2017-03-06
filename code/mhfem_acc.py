#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d 
from scipy.linalg import solve_banded 

import sys 

from exactDiff import * 

class MHFEM:

	def __init__(self, xe, Sigmaa, Sigmat, BCL=0, BCR=1):
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
		mu2f = interp1d(self.xe, mu2, kind='cubic') 
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
				self.A[1,i+2] = 3*beta*mu2f(self.x[i+2])

				# second upper (phi_i+3/2)
				self.A[0,i+3] = -beta*mu2f(self.x[i+3])

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
			print('left boundary condition not defined')
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
			print('right boundary condition not defined')
			sys.exit()

	def getEdges(self, phi):

		# get edge values 
		phiEdge = np.zeros(self.N)

		ii = 0 
		for i in range(0, self.n, 2):

			phiEdge[ii] = phi[i]

			ii += 1 

		return phiEdge

	def getCenters(self, phi):

		# get center values 
		phiCent = np.zeros(self.N-1) 

		ii = 0
		for i in range(1, self.n, 2):

			phiCent[ii] = phi[i]

			ii += 1 

		return phiCent

	def solve(self, q, qq, CENT=0):
		''' Compute phi = A^-1 q with banded solver 
			Inputs:
				q: cell edged array of source terms 
					uses cell average 
				qq: cell edged array of first moment of source 
				CENT: return phi on cell edges or edges and centers 
					0: edges only 
					1: centers only 
					2: edges and centers
		''' 

		ii = 0 # store iterations of q 
		b = np.zeros(self.n) # store source vector 
		# set odd equations to the source, leave even as zero 
		for i in range(1, self.n, 2):

			b[i] = (q[ii] + q[ii+1])/2 * (self.x[i+1] - self.x[i-1])

			ii += 1 

		# set even equations to use first moment of q 
		ii = 0 
		for i in range(1, self.n-2, 2):

			# alpha = 2/(self.Sigmat(self.x[i]) * (self.x[i+1] - self.x[i-1]))
			# alpha1 = 2/(self.Sigmat(self.x[i+2]) * (self.x[i+3] - self.x[i+1]))

			# b[i+1] = .5*(.5*alpha*(qq[ii] + qq[ii+1]) - .5*alpha1*(qq[ii+1] + qq[ii+2]))

			beta = 2/(self.Sigmat(self.x[i]))
			beta1 = 2/(self.Sigmat(self.x[i+2])) 

			b[i+1] = .5*(beta1*(qq[ii+2] + qq[ii + 1])/2 - beta*(qq[ii+1] + qq[ii])/2)

			ii += 1 

		# set boundary b 
		beta1 = 2/(self.Sigmat(self.x[1]))
		b[0] = beta1/2*(qq[1] + qq[0])/2 

		betaN = 2/(self.Sigmat(self.x[-2]))
		b[-1] = betaN/2*(qq[-1] + qq[-2])/2 

		# plt.plot(b)
		# plt.show()

		# solve for flux 
		# solve banded matrix 
		phi = solve_banded((2,2), self.A, b)

		# check solution 
		# self.checkSolution(phi, self.mu2f(self.x), q)

		if (CENT == 0): # return edges only 

			return self.xe, self.getEdges(phi)

		elif (CENT == 1): # return centers only 

			return self.xc, self.getCenters(phi)

		else: # return edges and centers 

			return self.x, phi 

	def checkSolution(self, phi, mu2, q):

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

		# check boundary 
		print(Jl[0]/phiEdge[0])
		print(Jr[-1]/phiEdge[-1])

		cont = np.zeros(self.N - 2)

		for i in range(1, self.N - 1):

			cont[i-1] = np.fabs((Jl[i] - Jr[i-1])/Jr[i-1])

		plt.figure()
		plt.semilogy(self.xe[1:-1], cont, '-o')

		# conservation 
		balance = np.zeros(self.N-1)
		for i in range(self.N-1):

			h = self.xe[i+1] - self.xe[i] 

			qq = .5*(q[i] + q[i+1]) * h

			balance[i] = Jr[i] - Jl[i] + self.Sigmaa(self.xc[i])*phiCent[i]*h - qq

			balance[i] /= qq

		plt.figure()
		plt.semilogy(self.xc, np.fabs(balance), '-o')
		plt.show()


if __name__ == '__main__':

	eps = 1e-3
	Sigmaa = .1*eps 
	Sigmat = .83/eps

	xb = 1

	Q = 1 * eps 

	BCL = 1
	BCR = 1 

	N = 100 # number of edges 
	xe = np.linspace(-xb, xb, N)
	mu2 = np.ones(N)/3 
	mhfem = MHFEM(xe, lambda x: Sigmaa, lambda x: Sigmat, BCL, BCR)
	mhfem.discretize(mu2, np.ones(N)/2)
	x, phi = mhfem.solve(np.ones(N)*Q, np.zeros(N), CENT=2)

	phiEdge = mhfem.getEdges(phi)
	phiCent = mhfem.getCenters(phi)

	xEdge = mhfem.getEdges(x)
	xCent = mhfem.getCenters(x)

	phiAvg = np.zeros(len(phiCent))

	for i in range(1, len(phiEdge)):

		phiAvg[i-1] = .5*(phiEdge[i] + phiEdge[i-1])

	print(phiCent/phiAvg)

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

	# plt.plot(x_ex, phi_ex(x_ex), '--')
	# plt.plot(x, phi, '-o')
	# # plt.plot(x, np.fabs(phi - phi_ex(x)), '-o')
	# # plt.yscale('log')
	# plt.show()

	# check order of convergence 
	# N = np.array([20, 40, 80, 160])

	# err = np.zeros(len(N))
	# for i in range(len(N)):

	# 	mh = MHFEM(np.linspace(0, xb, N[i]), lambda x: Sigmaa, 
	# 		lambda x: Sigmat, BCL=0, BCR=1)

	# 	mh.discretize(np.ones(N[i])/3, np.ones(N[i])/2)

	# 	x, phi = mh.solve(np.ones(N[i])*Q, CENT=1)

	# 	phif = interp1d(x, phi)

	# 	# err[i] = np.linalg.norm(phi - phi_ex(x), 2)
	# 	err[i] = np.fabs(phif(xb/2) - phi_ex(xb/2))

	# fit = np.polyfit(np.log(1/(N)), np.log(err), 1)

	# print(fit[0])

	# plt.loglog(1/(N), np.exp(fit[1]) * (1/(N))**fit[0], '-o')
	# plt.loglog(1/(N), err, '-o')
	# plt.show()




