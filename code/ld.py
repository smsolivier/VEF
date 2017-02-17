#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class LD:

	def __init__(self, x, n, Sigmaa, Sigmat, q):

		self.N = np.shape(x)[0] # number of cell edges 
		self.n = n # number of discrete ordinates 

		self.x = x 

		assert(n%2 == 0) # assert n is even 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x)
		self.q = q 

		# store all psis at each cell edge 
		self.psiL = np.zeros((n, self.N)) # mu < 0  
		self.psiR = np.zeros((n, self.N)) # mu > 0 

		self.phi = np.zeros(self.N) # store flux 

		# generate mu's 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

		# split into positive and negative 
		# ensure +/- pairs are matched by index, ie 0 --> +- first mu 
		self.muR = self.mu[self.mu > 0] # get right moving mu's 
		self.muL = -1*self.muR # get left moving mu's 

		# use symmetry to set wL 
		self.wR = self.w[self.mu > 0] # right moving weights 
		self.wL = np.copy(self.wR) # left moving weights 

	def sweepLR(self, phi):
		''' sweep left to right ''' 

		# loop through positive angles 
		for i in range(int(self.n/2)):

			

	def integratePsi(self):
		''' use guass legendre quadrature points to integrate psi ''' 

		phi = np.zeros(self.N)

		# loop through angles
		for i in range(int(self.n/2)):

			phi += self.psiL[i,:] * self.wL[i] + self.psiR[i,:] * self.wR[i]

		return phi 

	def getEddington(self):
		''' compute Eddington factor ''' 

		# compute <mu^2> 
		top = 0 # store int mu^2 psi dmu 
		# loop through angles
		for i in range(int(self.n/2)):

			top += self.muR[i]**2*self.psiL[i,:] * self.wL[i] + \
				self.muR[i]**2*self.psiR[i,:] * self.wR[i]

		mu2 = top/self.integratePsi()

		return mu2 

	def sourceIteration(self, tol, PLOT=False):
		''' lag RHS of transport equation and iterate until flux converges ''' 

		it = 0 # store number of iterations 
		phi = np.zeros(self.N) # store flux at each spatial location  
		edd = np.zeros(self.N) # store eddington factor 

		self.phiCon = [] 
		self.eddCon = [] 

		while (True):

			phi_old = np.copy(phi) # store old flux 
			edd_old = np.copy(edd) # store old edd 

			phi = self.sweep(phi) # update flux 
			edd = self.getEddington()

			self.phiCon.append(np.linalg.norm(phi - phi_old, 2)/np.linalg.norm(phi, 2))
			self.eddCon.append(np.linalg.norm(edd - edd_old, 2)/np.linalg.norm(edd, 2))

			# check for convergence 
			if (np.linalg.norm(phi - phi_old, 2)/np.linalg.norm(phi, 2) < tol):

				break 

			# update iteration count 
			it += 1 

		print('Number of iterations =', it) 

		if (PLOT):
			for i in range(int(self.n/2)):
				plt.plot(self.x, self.psiL[i,:], label=str(self.muL[i]))
				plt.plot(self.x, self.psiR[i,:], label=str(self.muR[i]))

			plt.legend(loc='best')
			plt.show()

		# return spatial locations, flux and number of iterations 
		return self.x, phi, it 