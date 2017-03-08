#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Direct:

	def __init__(self, xe, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		self.N = np.shape(xe)[0] - 1 # number of cell centers 
		self.Ne = np.shape(xe)[0] # number of cell edges 
		
		# boundary conditions 
		self.BCL = BCL
		self.BCR = BCR 

		self.h = np.zeros(self.N) # cell widths at cell center 
		self.xc = np.zeros(self.N) # cell centered locations 
		self.xe = xe # cell edged locations 

		# get cell centers and cell widths 
		for i in range(1, self.Ne):


			self.xc[i-1] = .5*(xe[i] + xe[i-1]) # get cell centers 
			self.h[i-1] = xe[i] - xe[i-1] # cell widths 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x) 
		self.q = q 

		# angles for S2 
		self.mu, self.w = np.polynomial.legendre.leggauss(2) 

	def discretize(self):

		# initialize matrix 
		A = np.zeros((2*self.Ne, 2*self.Ne))

		# interior cells 
		# psi_+ 

		ii = 0 # track which cell center 
		for i in range(2, 2*self.Ne - 2, 2):

			# cell centered properties 
			Sigmat = self.Sigmat(self.xc[ii])
			Sigmas = self.Sigmas(self.xc[ii])
			h = self.h[ii] 

			# psi_+,i-1/2 
			A[i,i-2] = -self.mu[1] + Sigmat*h/2 - Sigmas*h/4 

			# psi_-,i-1/2 
			A[i,i-1] = -Sigmas*h/4 

			# psi_+,i+1/2 
			A[i,i] = self.mu[1] + Sigmat*h/2 - Sigmas*h/4 

			# psi_-,i+1/2 
			A[i,i+1] = -Sigmas*h/4 

			ii += 1

		# psi_- 

		ii = 0 
		for i in range(3, 2*self.Ne-1, 2):

			Sigmat = self.Sigmat(self.xc[ii]) 
			Sigmas = self.Sigmas(self.xc[ii])
			h = self.h[ii] 

			# psi_+,i-1/2 
			A[i,i-3] = -Sigmas*h/4 

			# psi_-,i-1/2 
			A[i,i-2] = np.fabs(self.mu[0]) + .5*Sigmat*h - Sigmas*h/4 

			# psi_+,i+1/2 
			A[i,i-1] = -Sigmas*h/4 

			# psi_-,i+1/2 
			A[i,i] = -np.fabs(self.mu[0]) + Sigmat*h/2 - Sigmas*h/4 

			ii += 1 

		plt.imshow(A, interpolation='none')
		plt.colorbar()
		plt.show()

N = 10 
xb = 10 

Sigmaa = lambda x: .1 
Sigmat = lambda x: .83 

q = np.ones(N)

x = np.linspace(0, xb, N)

direct = Direct(x, Sigmaa, Sigmat, q)

direct.discretize()	