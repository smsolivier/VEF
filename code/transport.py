#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import Timer 

import os 
import sys 

import shutil 

''' General Discrete Ordinates class 
	Provides common variable names and helpful functions 
	Parent class for all spatial discretization methods and acceleration methods 
''' 

class Transport:
	''' general transport initialization ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):
		''' Inputs:
				xe: array of cell edges
				n: number of discrete ordinates 
				Sigmaa: absorption XS (function)
				Sigmat: total XS (function)
				q: fixed source (function of x and mu)  
				BCL: left boundary condition 
					0: reflecting 
					1: vacuum 
				BCR: right boundary condition
					0: reflecting 
					1: vacuum 
		''' 

		self.name = None # store name of method

		self.N = np.shape(xe)[0] - 1 # number of cell centers 
		self.Ne = np.shape(xe)[0] # number of cell edges 
		self.n = n # number of discrete ordinates 
		
		# left and right boundary conditions 
		self.BCL = BCL
		self.BCR = BCR 

		self.h = np.zeros(self.N) # cell widths at cell center 
		self.xc = np.zeros(self.N) # cell centered locations 
		self.xe = xe # cell edge array 

		# use cell edge in source iteration, self.x is returned in source iteration  
		self.x = self.xe 

		self.xb = xe[-1] # end of domain 

		# get cell centers and cell widths 
		for i in range(1, self.Ne):

			self.xc[i-1] = .5*(xe[i] + xe[i-1]) # get cell centers 
			self.h[i-1] = xe[i] - xe[i-1] # get cell widths 

		assert(n%2 == 0) # assert number of discrete ordinates is even 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x)

		# initialize psi and phi 
		self.psi = np.zeros((self.n, self.Ne)) # cell edged flux 
		self.phi = np.zeros(self.Ne) # store flux 

		# generate mu's, mu is arranged negative to positive 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

		# build q array from function 
		self.q = np.zeros((self.n, self.N))
		for i in range(self.n):

			for j in range(self.N):

				self.q[i,j] = q(self.xc[j], self.mu[i])

	def setMMS(self):
		''' setup MMS q 
			force phi = sin(pi*x/xb)
		''' 

		# mms solution 
		self.phi_mms = lambda x: np.sin(np.pi*x/self.xb)

		# ensure correct BCs 
		self.BCL = 1 
		self.BCR = 1 

		# loop through all angles 
		for i in range(self.n):

			# loop through space 
			for j in range(self.N):

				# set q 
				self.q[i,j] = self.mu[i]*np.pi/self.xb * \
					np.cos(np.pi*self.xc[j]/self.xb) + (self.Sigmat(self.xc[j]) - 
						self.Sigmas(self.xc[j]))*np.sin(np.pi*self.xc[j]/self.xb)

	def zeroMoment(self, psi):
		''' use guass quadrature points to integrate psi ''' 

		phi = np.zeros(np.shape(psi)[1])

		for i in range(self.n):

			phi += psi[i,:] * self.w[i] 

		return phi 

	def firstMoment(self, psi):
		''' use guass quadrature to integrate mu psi ''' 

		J = np.zeros(np.shape(psi)[1])

		for i in range(self.n):

			J += self.mu[i] * psi[i,:] * self.w[i] 

		return J 

	def getEddington(self, psi):
		''' compute <mu^2> ''' 

		phi = self.zeroMoment(psi)

		# Eddington factor 
		mu2 = np.zeros(np.shape(psi)[1]) 

		top = 0 
		for i in range(self.n):

			top += self.mu[i]**2 * psi[i,:] * self.w[i] 

		mu2 = top/phi 

		return mu2

	def pointConvergence(self, phi, phi_old):
		''' check if all points converged ''' 

		N = np.shape(phi)[0] # number of points 

		x = np.zeros(N)

		for i in range(N):

			x[i] = np.fabs(phi[i] - phi_old[i])

		return x 

	def sourceIteration(self, tol, maxIter=50, PLOT=None):
		''' lag RHS of transport equation and iterate until flux converges 
			PLOT: if true, makes a video of the flux converging 
		''' 

		it = 0 # store number of iterations 

		tt = Timer.timer()

		self.phiConv = [] # store convergence criterion for flux 

		if (PLOT != None):

			if (os.path.isdir(PLOT)):

				shutil.rmtree(PLOT)

			os.makedirs(PLOT)

			if (os.path.isfile(PLOT + '.mp4')):

				os.remove(PLOT + '.mp4')

		while (True):

			# check if max reached 
			if (it == maxIter):

				print('\n--- WARNING: maximum number of source iterations reached ---\n')
				break 

			# store old flux 
			phi_old = np.copy(self.phi) 

			if (PLOT != None):
				
				plt.figure()
				plt.plot(self.x, self.phi, '-o')
				# plt.ylim(0, 1.2)
				plt.title('Number of Iterations = ' + str(it))
				plt.savefig(PLOT + '/' + str(it) + '.png')
				plt.close()

			self.phi = self.sweep(phi_old) # update flux 

			self.phiConv.append(np.linalg.norm(self.phi - phi_old, 2)/
				np.linalg.norm(self.phi, 2))

			# check for convergence 
			if (self.phiConv[-1] < tol):

				break 

			# update iteration count 
			it += 1

			print(it, end='\r')

		print('Number of iterations =', it, end=', ') 
		tt.stop()

		if (PLOT != None):

			# make video 
			os.system('ffmpeg -f image2 -r 2 -i ' + PLOT + '/%d.png -b 320000k ' + PLOT + '.mp4')

		# return spatial locations, flux and number of iterations 
		return self.x, self.phi, it 

