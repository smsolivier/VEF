#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fem import * 
from mhfem_diff import * 
from fv_diff import * 

import mhfem_acc as mhfemacc 
import fem as fem

class Transport:
	''' Diamond (Crank Nicolson) differenced transport 
		mu dpsi/dx + sigmat psi = sigmas/2 phi + Q/2 
		parent class containing:
			left to right sweeping 
			right to left sweeping 
			gauss legendre psi integrator 
			source iteration 
		must supply the sweep(phi) function in the inherited class for 
			SI to work 
		assumes uniform source, q 
	'''  

	def __init__(self, N, n, Sigmaa, Sigmat, q, xb=1):

		self.N = N # number of cell edges  
		self.n = n # number of discrete ordinates 

		assert(n%2 == 0) # assert n is even 

		self.xb = xb 

		self.h = xb/N 

		self.x = np.linspace(0, xb, N) 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = Sigmat - Sigmaa
		self.q = q 

		# store all psis at each cell edge 
		self.psiL = np.zeros((n, N)) # mu < 0  
		self.psiR = np.zeros((n, N)) # mu > 0 

		self.phi = np.zeros(N) # store flux 

		print('c =', self.Sigmas/Sigmat)

		# generate mu's 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

		# split into positive and negative 
		# ensure +/- pairs are matched by index, ie 0 --> +- first mu 
		self.muR = self.mu[self.mu > 0] # get right moving mu's 
		self.muL = -1*self.muR # get left moving mu's 

		# use symmetry to set wL 
		self.wR = self.w[self.mu > 0] # right moving weights 
		self.wL = np.copy(self.wR) # left moving weights 
		
	def sweepRL(self, phi):
		''' sweep right to left ''' 

		# loop through negative angles  
		for i in range(int(self.n/2)):

			# spatial loop from right to left 
			for j in range(self.N-2, -1, -1):

				# rhs 
				b = self.Sigmas/4*(phi[j] + phi[j+1]) + self.q/2

				self.psiL[i,j] = b*self.h - (.5*self.Sigmat*self.h - 
					np.fabs(self.muL[i]))*self.psiL[i,j+1]

				self.psiL[i,j] /= .5*self.Sigmat*self.h + np.fabs(self.muL[i])

	def sweepLR(self, phi):
		''' sweep left to right ''' 

		# loop through positive angles 
		for i in range(int(self.n/2)):

			# store discretized rhs 
			

			# spatial loop from left to right 
			for j in range(1, self.N):

				# rhs 
				b = self.Sigmas/4*(phi[j] + phi[j-1]) + self.q/2 

				self.psiR[i,j] = b*self.h - (.5*self.Sigmat*self.h - 
					self.muR[i])*self.psiR[i,j-1]

				self.psiR[i,j] /= .5*self.Sigmat*self.h + self.muR[i] 

	def integratePsi(self):
		''' use guass legendre quadrature points to integrate psi ''' 

		phi = np.zeros(self.N)

		# loop through angles
		for i in range(int(self.n/2)):

			phi += self.psiL[i,:] * self.wL[i] + self.psiR[i,:] * self.wR[i]

		return phi 

	def getEddington(self):

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
		phi = np.zeros(self.N) 

		while (True):

			phi_old = np.copy(phi) # store old flux 

			phi = self.sweep(phi) # update flux 

			# check for convergence 
			if (np.linalg.norm(phi - phi_old, 2) < tol):

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

class Sn(Transport):
	''' inherits discretization initialization, sweepLR, sweepRL, sourceIteration 
			from Transport class 
	''' 

	def sweep(self, phi):
		''' sweep with reflecting left bc and vacuum right bc ''' 

		self.psiL[:,-1] = 0 # vacuum bc, no left moving entering left side 
		self.sweepRL(phi) # sweep right to left  

		# set psiR boundary 
		self.psiR[:,0] = self.psiL[:,0] # reflecting bc 
		self.sweepLR(phi) # sweep left to right  

		# return flux 
		return self.integratePsi() 


class DSA(Transport):
	''' inherits discretization initialization, sweepLR, sweepRL, sourceIteration 
			from Transport class 
	''' 

	# override Transport initialization 
	def __init__(self, N, n, Sigmaa, Sigmat, q, xb):

		# call Transport initialization 
		Transport.__init__(self, N, n, Sigmaa, Sigmat, q, xb)

		# create fem object on top of standard Transport initialization 
		self.fem = mhfemacc.MHFEM(self.x, np.ones(self.N)/3, lambda x: self.Sigmaa, 
			lambda x: self.Sigmat, xb=self.xb, BCL=0, BCR=1)

	def sweep(self, phi):

		# transport sweep 
		self.psiL[:,-1] = 0 # vacuum bc, no left moving entering left side 
		self.sweepRL(phi) # sweep right to left  

		# set psiR boundary 
		self.psiR[:,0] = self.psiL[:,0] # reflecting bc 
		self.sweepLR(phi) # sweep left to right  

		# compute phi^l+1/2 
		phihalf = self.integratePsi()

		# DSA step 
		x, f = self.fem.solve(self.Sigmas*(phihalf - phi))

		# compute new flux 
		return phihalf + f 

class muAccel(Transport):
	''' inherits discretization initialization, sweepLR, sweepRL, sourceIteration, 
			getEddingtion from Transport class 
		Eddington acceleration sweep method 
	''' 

	def sweep(self, phi):

		# transport sweep 
		self.psiL[:,-1] = 0 # vaccuum bc, no left moving entering left side 
		self.sweepRL(phi) # sweep right to left 

		# set psiR boundary 
		self.psiR[:,0] = self.psiL[:,0] # reflecting bc 
		self.sweepLR(phi) # sweep left to right 

		# compute phi^l+1/2 
		self.phihalf = self.integratePsi()

		mu2 = self.getEddington() # get <mu^2> 

		# create MHFEM object 
		sol = mhfemacc.MHFEM(self.x, mu2, lambda x: self.Sigmaa, 
			lambda x: self.Sigmat, BCL=0, BCR=1)

		x, phi = sol.solve(self.q*np.ones(self.N))

		return phi 


if __name__ == '__main__':

	N = 200 # number of edges 
	Sigmat = 1 
	c = .9 # ratio of Sigmas to Sigmat 
	Sigmaa = Sigmat*(1 - c) 
	q = 1
	xb = 1

	tol = 1e-6 

	n = 8

	sn = Sn(N, n, Sigmaa, Sigmat, q, xb=xb)
	mu = muAccel(N, n, Sigmaa, Sigmat, q, xb=xb)
	# dsa = DSA(N, n, Sigmaa, Sigmat, q, xb=xb)
	# diff = finiteVolume(N, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2)
	# diff = FEM(N, Sigmaa, Sigmat, xb=xb)
	# diff = MHFEM(N-1, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2, EDGE=1)
	diff = FEM(np.linspace(0, xb, N), np.ones(N)/3, lambda x: Sigmaa, 
		lambda x: Sigmat, BCL=0, BCR=1)

	x, phi, it = sn.sourceIteration(tol)
	xmu, phimu, itmu = mu.sourceIteration(tol)
	# xdsa, phidsa, itdsa = dsa.sourceIteration(tol)
	xdiff, phidiff = diff.solve(np.ones(N)*q)

	plt.plot(xmu, mu.getEddington())
	plt.show()

	plt.plot(x, phi, label='S'+str(n)+' SI')
	plt.plot(xmu, mu.phihalf, label='S'+str(n)+' mu')
	# plt.plot(xdsa, phidsa, label='S' + str(n) + ' DSA')
	plt.plot(xdiff, phidiff, label='Diffusion')
	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('$\phi$')
	plt.show()

