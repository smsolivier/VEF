#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_acc import * 

import Timer 

class LD:
	''' Linear Discontinuous Galerkin Sn ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):
		''' Inputs:
				xe: cell edges 
				n: number of discrete ordinates 
				Sigmaa: absorption XS (function)
				Sigmat: total XS (function)
				q: fixed source array of mu and cell edge spatial dependence 
		''' 

		self.N = np.shape(xe)[0] - 1 # number of cell centers 
		self.Ne = np.shape(xe)[0] # number of cell edges 
		self.n = n # number of discrete ordinates 
		
		self.BCL = BCL
		self.BCR = BCR 

		self.x = np.zeros(self.N) # cell centered locations 
		self.h = np.zeros(self.N) # cell widths at cell center 

		self.xe = xe # cell edge array 
		self.xb = xe[-1] # end of domain 

		for i in range(1, self.Ne):

			self.x[i-1] = .5*(xe[i] + xe[i-1]) # get cell centers 
			self.h[i-1] = xe[i] - xe[i-1] # get cell widths 

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

		self.psi = np.zeros((self.n, self.Ne)) # cell edged flux 

		self.phi = np.zeros(self.N) # store flux 

		# generate mu's, mu is arranged negative to positive 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

	def setMMS(self):
		''' setup MMS q 
			force phi = sin(pi*x/xb)
		''' 

		# ensure correct BCs 
		self.BCL = 1 
		self.BCR = 1 

		# loop through all angles 
		for i in range(self.n):

			# loop through space 
			for j in range(self.Ne):

				self.q[i,j] = self.mu[i]*np.pi/self.xb * \
					np.cos(np.pi*self.xe[j]/self.xb) + (self.Sigmat(self.xe[j]) - 
						self.Sigmas(self.xe[j]))*np.sin(np.pi*self.xe[j]/self.xb)

	def fullSweep(self, phi):
		''' sweep left to right or right to left depending on boundary conditions ''' 

		if (self.BCL == 0 and self.BCR == 1): # left reflecting 

			self.sweepRL(phi)
			self.sweepLR(phi)

		elif (self.BCR == 0 and self.BCL == 1): # right reflecting 

			self.sweepLR(phi)
			self.sweepRL(phi)

		else:

			self.sweepLR(phi)
			self.sweepRL(phi)

	def sweep(self, phi):
		''' unaccelerated sweep ''' 

		# sweep left to right or right to left first depending on BCs 
		self.fullSweep(phi)

		# convert to edge values 
		psi = self.edgePsi()

		# get edge flux 
		phi = self.integratePsi(psi)

		return phi 

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
				b[0] = self.Sigmas(self.x[j])*h/4*phi[j] + self.q[i,j]*h/4 # left 
				if (j == 0): # boundary condition 

					# default to vacuum 

					if (self.BCL == 0): # reflecting 

						b[0] += self.mu[i]*self.psiL[self.mu == -self.mu[i],0]

				else: # normal sweep 

					b[0] += self.mu[i]*self.psiR[i,j-1] # upwind term 

				b[1] = self.Sigmas(self.x[j])*h/4*phi[j+1] + self.q[i,j+1]*h/4 # right 

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
				b[0] = self.Sigmas(self.x[j])*h/4*phi[j] + self.q[i,j]*h/4 # left 
				b[1] = self.Sigmas(self.x[j])*h/4*phi[j+1] + self.q[i,j+1]*h/4 # right 

				if (j == self.N-1): # boundary condition 
					
					# default to vacuum 

					if (self.BCR == 0): # reflecting 

						b[1] -= self.mu[i]*self.psiR[self.mu == -self.mu[i],-1]
 
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
				if (self.BCL == 0): # reflecting 

					psi[i,0] = self.psiL[self.mu == -self.mu[i],0] 

				elif (self.BCL == 1):

					psi[i,0] = 0 

				for j in range(self.N):

					psi[i,j+1] = self.psiR[i,j] # psi_j+1/2 = psi_i,R  

			else: # negative angles 

				# set boundary values 
				if (self.BCR == 0): # reflecting 

					psi[i,-1] = self.psiR[self.mu == -self.mu[i],-1]

				elif (self.BCR == 1): # vacuum 

					psi[i,-1] = 0 

				for j in range(self.N-1, -1, -1):

					psi[i,j] = self.psiL[i,j] # psi_j-1/2 = psi_i,L

		self.psi = psi

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

		tt = Timer.timer()

		while (True):

			# store old flux 
			phi_old = np.copy(phi) 

			phi = self.sweep(phi_old) # update flux 

			# check for convergence 
			if (np.linalg.norm(phi - phi_old, 2)/np.linalg.norm(phi, 2) < tol):

				break 

			# update iteration count 
			it += 1 

		print('Number of iterations =', it, end=', ') 
		tt.stop()

		# return spatial locations, flux and number of iterations 
		return self.xe, phi, it 

class Eddington(LD):
	''' Eddington accelerated ''' 

	def sweep(self, phi):

		self.fullSweep(phi)

		psi = self.edgePsi()

		# compute eddington factor 
		mu2 = self.getEddington(psi)

		# generate boundary eddington for consistency between drift and transport 
		top = 0 
		for i in range(self.n):

			top += np.fabs(self.mu[i])*psi[i,:] * self.w[i] 

		B = top/self.integratePsi(psi)*2

		# create MHFEM object 
		sol = MHFEM(self.xe, mu2, self.Sigmaa, self.Sigmat, B, BCL=self.BCL, BCR=self.BCR)

		# solve for phi 
		x, phi = sol.solve(self.integratePsi(self.q)/2)

		return phi # return accelerated flux 

if __name__ == '__main__':

	N = 40
	xb = 2
	x = np.linspace(0, xb, N)
	Sigmaa = lambda x: .1 
	Sigmat = lambda x: .83

	n = 16

	q = np.ones((n, N)) 

	tol = 1e-6 

	ld = LD(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)
	# ld.setMMS()
	# ed = Eddington(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)

	x, phi, it = ld.sourceIteration(tol)

	# xe, phie, ite = ed.sourceIteration(tol)

	plt.plot(x, phi, label='LD')
	# plt.plot(xe, phie, label='LD Edd')
	plt.legend(loc='best')
	plt.show()