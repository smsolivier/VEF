#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from transport import * # general transport class 

class LD(Transport):
	''' Linear Discontinuous Galerkin spatial discretization of Sn 
		Inherits functions from transport.py 
	''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call transport initialization 
		Transport.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		# create LD specific variables 
		# store LD left and right discontinuous points 
		# psi = .5*(psiL + psiR) 
		# store for all mu and each cell center 
		self.psiL = np.zeros((self.n, self.N)) # LD left point  
		self.psiR = np.zeros((self.n, self.N)) # LD right point 

		# store LD flux, cell centered 
		self.phiL = np.zeros(self.N) 
		self.phiR = np.zeros(self.N)

	def fullSweep(self, phiL, phiR):
		''' sweep left to right or right to left depending on boundary conditions ''' 

		if (self.BCL == 0 and self.BCR == 1): # left reflecting 

			self.sweepRL(phiL, phiR)
			self.sweepLR(phiL, phiR)

		elif (self.BCR == 0 and self.BCL == 1): # right reflecting 

			self.sweepLR(phiL, phiR)
			self.sweepRL(phiL, phiR)

		else:

			self.sweepLR(phiL, phiR)
			self.sweepRL(phiL, phiR)

	def sweep(self, phi):
		''' unaccelerated sweep ''' 

		phiL, phiR = self.ldRecovery(phi)

		# sweep left to right or right to left first depending on BCs 
		self.fullSweep(phiL, phiR)

		# convert to edge values 
		psi = self.edgePsi()

		# get edge flux 
		phi = self.zeroMoment(psi)

		return phi 

	def sweepLR(self, phiL, phiR):
		''' sweep left to right (mu > 0) ''' 

		A = np.zeros((2,2)) # store psiL, psiR coefficients 
		b = np.zeros(2) # store rhs 

		# loop through positive angles 
		for i in range(int(self.n/2), self.n):

			# loop from 1:N, assume BC already set 
			for j in range(0, self.N):

				h = self.h[j] # cell width 

				# left equation 
				A[0,0] = self.mu[i]/2 + self.Sigmat(self.xc[j])*h/2 # psiL 
				A[0,1] = self.mu[i]/2 # psiR 

				# right equation 
				A[1,0] = -self.mu[i]/2 # psiL
				A[1,1] = -self.mu[i]/2 + self.Sigmat(self.xc[j])*h/2 + self.mu[i] # psiR 

				# rhs 
				b[0] = self.Sigmas(self.xc[j])*h/4*phiL[j] + self.q[i,j]*h/4 # left 
				if (j == 0): # boundary condition 

					# default to vacuum 

					if (self.BCL == 0): # reflecting 

						b[0] += self.mu[i]*self.psiL[self.mu == -self.mu[i],0]

				else: # normal sweep 

					b[0] += self.mu[i]*self.psiR[i,j-1] # upwind term 

				b[1] = self.Sigmas(self.xc[j])*h/4*phiR[j] + self.q[i,j+1]*h/4 # right 

				ans = np.linalg.solve(A, b) # solve for psiL, psiR 

				# extract psis 
				self.psiL[i,j] = ans[0] 
				self.psiR[i,j] = ans[1] 

	def sweepRL(self, phiL, phiR):
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
				A[1,1] = -self.mu[i]/2 + self.Sigmat(self.xc[j])*h/2 # psiR 

				# right equation 
				A[0,0] = self.mu[i]/2 + self.Sigmat(self.xc[j])*h/2 - self.mu[i] # psiL 
				A[0,1] = self.mu[i]/2 

				# rhs 
				b[0] = self.Sigmas(self.xc[j])*h/4*phiL[j] + self.q[i,j]*h/4 # left 
				b[1] = self.Sigmas(self.xc[j])*h/4*phiR[j] + self.q[i,j+1]*h/4 # right 

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

	def ldRecovery(self, phi):
		''' Recover LD left and right values 
			use edge values for left and right LD values 
		'''

		phiL = np.zeros(self.N) # left flux 
		phiR = np.zeros(self.N) # right flux 

		for i in range(self.N):

			phiL[i] = phi[i] 

		for i in range(self.N):

			phiR[i] = phi[i+1] 

		return phiL, phiR 

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

class Eddington(LD):

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call LD initialization 
		LD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		# redefine phi to be on cell edges and centers 
		self.phi = np.zeros(2*self.Ne - 1) 

		# redefine x to be all points 
		self.x = np.sort(np.concatenate((self.xc, self.xe)))

		# create MHFEM object 
		self.mhfem = MHFEM(self.xe, self.Sigmaa, self.Sigmat, self.BCL, self.BCR)

	def ldRecovery(self, phi, OPT=1):
		''' Recover LD left and right values 
			OPT: determine which recovery scheme to use 
				0: use edges (calls useHalf)
				1: maintain average and slope (calls maintainSlopes)
		'''

		if (OPT == 0):

			phiL, phiR = self.useHalf(phi)

		elif (OPT == 1):

			phiL, phiR = self.maintainSlopes(phi) 

		return phiL, phiR 

	def maintainSlopes(self, phi):
		''' maintain cell center value and slopes ''' 

		phiL = np.zeros(self.N) # left flux 
		phiR = np.zeros(self.N) # right flux 

		# separate edges and centers 
		phiEdge = self.mhfem.getEdges(phi) # get edges from MHFEM object utility 
		phiCent = self.mhfem.getCenters(phi) # get centers from MHFEM object utility 

		for i in range(self.N):

			# phi_i+1/2 - phi_i-1/2 
			diff = .5*(phiEdge[i+1] - phiEdge[i]) 

			# recover left and right 
			phiL[i] = phiCent[i] - diff 

			phiR[i] = phiCent[i] + diff

		return phiL, phiR 

	def useHalf(self, phi):
		''' use phi_iL = phi_i-1/2, phi_iR = phi_i+1/2 ''' 

		phiL = np.zeros(self.N)
		phiR = np.zeros(self.N) 

		phiEdge = self.mhfem.getEdges(phi) # get edges from MHFEM object utility 

		for i in range(self.N):

			phiL[i] = phiEdge[i] 
			phiR[i] = phiEdge[i+1] 

		return phiL, phiR 

	def sweep(self, phi):

		# get LD left and right fluxes 
		phiL, phiR = self.ldRecovery(phi, OPT=0)

		self.fullSweep(phiL, phiR) # transport sweep, BC dependent ordering 

		psi = self.edgePsi() # get edge values of psi 

		# compute eddington factor 
		mu2 = self.getEddington(psi)

		# generate boundary eddington for consistency between drift and transport 
		top = 0 
		for i in range(self.n):

			top += np.fabs(self.mu[i])*psi[i,:] * self.w[i] 

		B = top/self.zeroMoment(psi)

		# discretize MHFEM with mu^2 and B 
		self.mhfem.discretize(mu2, B)

		# solve for phi, get edges and centers 
		x, phi = self.mhfem.solve(self.zeroMoment(self.q)/2, CENT=2)

		return phi # return accelerated flux 

	def sourceIteration(self, tol):

		x, phi, it = LD.sourceIteration(self, tol)

		return self.xe, self.mhfem.getEdges(phi), it 

if __name__ == '__main__':

	N = 40
	xb = 2
	x = np.linspace(0, xb, N)
	Sigmaa = lambda x: .1 
	Sigmat = lambda x: .83

	n = 16

	q = np.ones((n, N)) 

	tol = 1e-10

	ld = LD(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)
	# ld.setMMS()
	ed = Eddington(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)

	x, phi, it = ld.sourceIteration(tol)

	xe, phie, ite = ed.sourceIteration(tol)

	plt.plot(x, phi, label='LD')
	plt.plot(xe, phie, label='LD Edd')
	plt.legend(loc='best')
	plt.show()