#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sn as DD

from mhfem_acc import * 

class LD:
	''' Linear Discontinuous Galerkin Sn ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, ACCEL=True):
		''' Inputs:
				xe: cell edges 
				n: number of discrete ordinates 
				Sigmaa: absorption XS (function)
				Sigmat: total XS (function)
				q: fixed source (cell edged array)
		''' 

		self.N = np.shape(xe)[0] - 1 # number of cell centers 
		self.n = n # number of discrete ordinates 
		self.ACCEL = ACCEL

		self.x = np.zeros(self.N) # cell centered locations 
		self.h = np.zeros(self.N) # cell widths at cell center 

		self.xe = xe # cell edge array 

		for i in range(1, self.N+1):

			self.x[i-1] = .5*(xe[i] + xe[i-1])
			self.h[i-1] = xe[i] - xe[i-1] 

		assert(n%2 == 0) # assert n is even 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x)
		self.q = q 

		# store LD left and right discontinuous points 
		# psi = .5*(psiL + psiR) 
		# store for all mu and each cell center 
		self.psiL = np.zeros((self.n, self.N)) # LD left point  
		self.psiR = np.zeros((self.n, self.N)) # LD right point 

		self.psi = np.zeros((self.n, self.N+1)) # cell edged flux 

		self.phi = np.zeros(self.N) # store flux 

		# generate mu's, mu is arranged negative to positive 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

	def sweep(self, phi):

		# sweep for psi 
		self.sweepRL(phi)
		self.sweepLR(phi)

		psi = self.edgePsi()

		if (self.ACCEL == False):

			# get flux 
			phi = self.integratePsi(psi)

			return phi 

		elif (self.ACCEL == True):

			# compute eddington factor 
			mu2 = self.getEddington(psi)

			top = 0 
			for i in range(self.n):

				top += np.fabs(self.mu[i])*psi[i,-1] * self.w[i] 

			mu2[-1] = top/self.integratePsi(psi)[-1]

			plt.plot(self.xe, mu2, '-o')
			plt.show()

			# create MHFEM object 
			sol = MHFEM(self.xe, mu2, self.Sigmaa, self.Sigmat, BCL=0, BCR=1)

			# solve for phi 
			x, phi = sol.solve(self.q)

			return phi # return accelerated flux 

	def sweepLR(self, phi):
		''' sweep left to right (mu > 0) ''' 

		A = np.zeros((2,2)) # store psiL, psiR coefficients 
		b = np.zeros(2) # store rhs 

		# loop through positive angles 
		for i in range(int(self.n/2), self.n):

			# loop from 1:N, assume BC already set 
			for j in range(0, self.N):

				h = self.h[j] # cell width 

				# left equation 
				A[0,0] = self.mu[i]/2 + self.Sigmat(self.x[j])*h/2 # psiL 
				A[0,1] = self.mu[i]/2 # psiR 

				# right equation 
				A[1,0] = -self.mu[i]/2 # psiL
				A[1,1] = -self.mu[i]/2 + self.Sigmat(self.x[j])*h/2 + self.mu[i] # psiR 

				# rhs 
				b[0] = self.Sigmas(self.x[j])*h/4*phi[j] + self.q[j]*h/4 # left 
				if (j == 0): # boundary condition 

					# reflecting 
					b[0] += self.mu[i]*self.psiL[self.mu == -self.mu[i],0]

				else: # normal sweep 
					b[0] += self.mu[i]*self.psiR[i,j-1] # upwind term 

				b[1] = self.Sigmas(self.x[j])*h/4*phi[j+1] + self.q[j+1]*h/4 # right 

				ans = np.linalg.solve(A, b) # solve for psiL, psiR 

				# extract psis 
				self.psiL[i,j] = ans[0] 
				self.psiR[i,j] = ans[1] 

	def sweepRL(self, phi):
		''' sweep right to left (mu < 0) ''' 

		A = np.zeros((2,2)) # store psiL, psiR coefficients 
		b = np.zeros(2) # store rhs 

		# loop through negative angles 
		for i in range(int(self.n/2)):

			# loop backwards from N-1:0, assume BC already set 
			for j in range(self.N-1, -1, -1):

				h = self.h[j] # cell width 

				# left equation 
				A[1,0] = -self.mu[i]/2 # psiL
				A[1,1] = -self.mu[i]/2 + self.Sigmat(self.x[j])*h/2 # psiR 

				# right equation 
				A[0,0] = self.mu[i]/2 + self.Sigmat(self.x[j])*h/2 - self.mu[i] # psiL 
				A[0,1] = self.mu[i]/2 

				# rhs 
				b[0] = self.Sigmas(self.x[j])*h/4*phi[j] + self.q[j]*h/4 # left 
				b[1] = self.Sigmas(self.x[j])*h/4*phi[j+1] + self.q[j+1]*h/4 # right 

				if (j == self.N-1): # boundary condition 
					b[1] += 0 # vacuum bc 
 
				else: # normal sweep
					b[1] -= self.mu[i]*self.psiL[i,j+1] # downwind term 

				ans = np.linalg.solve(A, b) # solve for psiL, psiR 

				# extract psis 
				self.psiL[i,j] = ans[0] 
				self.psiR[i,j] = ans[1] 

	def integratePsi(self, psi):
		''' use guass legendre quadrature points to integrate psi ''' 

		phi = np.zeros(self.N+1)

		for i in range(self.n):

			phi += psi[i,:] * self.w[i] 

		return phi 

	def edgePsi(self):
		''' get celled edge angular flux accounting for up/down winding ''' 

		psi = np.zeros((self.n, self.N+1)) # store cell edge psi 

		for i in range(self.n):

			if (self.mu[i] > 0): # positive angles 

				# set boundary values 
				psi[i,0] = self.psiL[self.mu == -self.mu[i],0] # reflecting 

				for j in range(self.N):

					psi[i,j+1] = self.psiR[i,j] # psi_j+1/2 = psi_i,R  

			else: # negative angles 

				# set boundary values 
				psi[i,-1] = 0 # vacuum 

				for j in range(self.N-1, -1, -1):

					psi[i,j] = self.psiL[i,j] # psi_j-1/2 = psi_i,L

		return psi 

	def getEddington(self, psi):
		''' compute <mu^2> ''' 

		phi = np.zeros(self.N + 1) # cell edge flux 

		for i in range(self.n):

			phi += psi[i,:] * self.w[i]

		# Eddington factor 
		mu2 = np.zeros(self.N+1) 

		top = 0 
		for i in range(self.n):

			top += self.mu[i]**2 * psi[i,:] * self.w[i] 

		mu2 = top/phi 

		return mu2

	def sourceIteration(self, tol):
		''' lag RHS of transport equation and iterate until flux converges ''' 

		it = 0 # store number of iterations 
		phi = np.zeros(self.N+1) # cell centered flux 

		while (True):

			# store old flux 
			phi_old = np.copy(phi) 

			phi = self.sweep(phi_old) # update flux 

			# check for convergence 
			if (np.linalg.norm(phi - phi_old, 2)/np.linalg.norm(phi, 2) < tol):

				break 

			# update iteration count 
			it += 1 

		print('Number of iterations =', it) 

		# return spatial locations, flux and number of iterations 
		return self.xe, phi, it 

N = 50
xb = 2
x = np.linspace(0, xb, N)
Sigmaa = lambda x: .1 
Sigmat = lambda x: .83 
q = np.ones(N) 

n = 16

ld = LD(x, n, Sigmaa, Sigmat, q)

sn = DD.Sn(x, n, Sigmaa, Sigmat, q)

mu = DD.muAccel(x, n, Sigmaa, Sigmat, q)

ld2 = LD(x, n, Sigmaa, Sigmat, q, False)

x, phi, it = ld.sourceIteration(1e-6)

# xsn, phisn, itsn = sn.sourceIteration(1e-6)

# xmu, phimu, itmu = mu.sourceIteration(1e-6)

x2, phi2, it2 = ld2.sourceIteration(1e-6)

plt.plot(x, phi, label='LD Edd')
# plt.plot(xsn, phisn, label='DD')
# plt.plot(xmu, phimu, '--', label='DD Edd')
plt.plot(x2, phi2, '--', label='LD')
plt.legend(loc='best')
plt.show()