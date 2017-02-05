#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fem import * 
from mhfem_diff import * 
from fv_diff import * 

class Transport:
	''' Diamond (Crank Nicolson) differences transport 
		mu dpsi/dx + sigmat psi = sigmas/2 phi + Q/2 
		parent class containing:
			left to right sweeping 
			right to left sweeping 
			source iteration 
		must supply the sweep(phi) function in the inherited class for 
			SI to work 
		assumes uniform source, q 
	'''  

	def __init__(self, N, Sigmaa, Sigmat, q, xb=1):

		self.N = N 

		self.xb = xb 

		self.h = xb/N 

		self.x = np.linspace(0, xb, N) 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = Sigmat - Sigmaa
		self.q = q 

		# store psi left and right 
		self.psiL = np.zeros(N) 
		self.psiR = np.zeros(N) 

		self.phi = np.zeros(N) # store flux 

		print('c =', self.Sigmas/Sigmat)

	def sweepLR(self, phi):
		''' sweep left to right ''' 

		for i in range(self.N-2, -1, -1):

			self.psiL[i] = (1 - .5*self.Sigmat*self.h)*self.psiL[i+1] + \
				self.Sigmas/4*self.h*(phi[i] + phi[i+1]) + self.q*self.h/2 

			self.psiL[i] /= 1 + .5*self.Sigmat*self.h

	def sweepRL(self, phi):
		''' sweep right to left ''' 

		for i in range(1, self.N):

			self.psiR[i] = (1 - .5*self.Sigmat*self.h)*self.psiR[i-1] + \
				self.Sigmas/4*self.h*(phi[i] + phi[i-1]) + self.q*self.h/2 

			self.psiR[i] /= 1 + .5*self.Sigmat*self.h 

	def sourceIteration(self, tol):

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

		return self.x, phi, it 

class Sn(Transport):
	''' inherits discretization initialization, sweepLR, sweepRL, sourceIteration 
			from Transport class 
	''' 

	def sweep(self, phi):
		''' sweep with reflecting left bc and vacuum right bc ''' 

		self.psiL[-1] = 0 # vacuum bc, no left moving entering left side 
		self.sweepLR(phi) # sweep left to right 

		# set psiR boundary 
		self.psiR[0] = self.psiL[0] # reflecting bc 
		self.sweepRL(phi) # sweep right to left 

		# return cell edge flux 
		return self.psiL + self.psiR # integrate psi 

class DSA(Transport):
	''' inherits discretization initialization, sweepLR, sweepRL, sourceIteration 
			from Transport class 
	''' 

	# override Transport initialization 
	def __init__(self, N, Sigmaa, Sigmat, q, xb):

		# call Transport initialization 
		Transport.__init__(self, N, Sigmaa, Sigmat, q, xb)

		# create fem object on top of standard Transport initialization 
		# self.fem = FEM(self.N, self.Sigmaa, self.Sigmat, self.xb)
		self.fem = MHFEM(self.N-1, self.Sigmaa, self.Sigmat, self.xb, BCL=0, BCR=2, EDGE=1)

	def sweep(self, phi):

		# transport sweep 
		self.psiL[-1] = 0 # vacuum bc, no left moving entering left side 
		self.sweepLR(phi) # sweep left to right 

		# set psiR boundary 
		self.psiR[0] = self.psiL[0] # reflecting bc 
		self.sweepRL(phi) # sweep right to left 

		# compute phi^l+1/2 
		phihalf = self.psiR + self.psiL 

		# DSA step 
		x, f = self.fem.solve(self.Sigmas*(phihalf - phi))

		# compute new flux 
		return phihalf + f 

if __name__ == '__main__':

	N = 200 # number of edges 
	Sigmat = 1 
	c = .9 # ratio of Sigmas to Sigmat 
	Sigmaa = Sigmat*(1 - c) 
	q = 1
	xb = 100 

	tol = 1e-6 

	sn = Sn(N, Sigmaa, Sigmat, q, xb=xb)
	dsa = DSA(N, Sigmaa, Sigmat, q, xb=xb)
	# diff = finiteVolume(N, lambda x: Sigmaa, lambda x: Sigmat, xb=xb, BCL=0, BCR=2)
	# diff = FEM(N, Sigmaa, Sigmat, xb=xb)
	diff = MHFEM(N-1, Sigmaa, Sigmat, xb=xb, BCL=0, BCR=2, EDGE=1)

	x, phi, it = sn.sourceIteration(tol)
	xdsa, phidsa, itdsa = dsa.sourceIteration(tol)
	xdiff, phidiff = diff.solve(np.ones(N)*q)

	plt.plot(x, phi, label=r'$S_2$')
	plt.plot(xdsa, phidsa, label=r'$S_2$ DSA')
	plt.plot(xdiff, phidiff, label='Diffusion')
	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('$\phi$')
	plt.show()

