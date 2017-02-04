#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Sn:

	def __init__(self, N, Sigmaa, Sigmat, q, xb=1, BCL=0, BCR=0):

		self.N = N 

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

	def sweep(self, phi):
		''' sweep with reflecting left bc and vacuum right bc ''' 

		self.psiL[-1] = 0 # vacuum bc, no left moving entering left side 
		self.sweepLR(phi) # sweep left to right 

		# set psiR boundary 
		self.psiR[0] = self.psiL[0] # reflecting bc 
		self.sweepRL(phi) # sweep right to left 

		return self.psiL + self.psiR # integrate psi 

	def sweepLR(self, phi):
		''' sweep left to right ''' 

		for i in range(N-2, -1, -1):

			self.psiL[i] = (1 - .5*self.Sigmat*self.h)*self.psiL[i+1] + \
				self.Sigmas/4*self.h*(phi[i] + phi[i+1]) + self.q*self.h/2 

			self.psiL[i] /= 1 + .5*self.Sigmat*self.h

	def sweepRL(self, phi):
		''' sweep right to left ''' 

		for i in range(1, N):

			self.psiR[i] = (1 - .5*self.Sigmat*self.h)*self.psiR[i-1] + \
				self.Sigmas/4*self.h*(phi[i] + phi[i-1]) + self.q*self.h/2 

			self.psiR[i] /= 1 + .5*self.Sigmat*self.h 

def sourceIteration(sweeper, tol):

	it = 0 # store number of iterations 
	phi = np.zeros(sweeper.N) 

	while (True):

		phi_old = np.copy(phi) # store old flux 

		phi = sweeper.sweep(phi) # update flux 

		# check for convergence 
		if (np.linalg.norm(phi - phi_old, 2) < tol):

			break 

		# update iteration count 
		it += 1 

	print('Number of iterations =', it) 

	return sweeper.x, phi, it 

N = 100
Sigmat = 1 
q = 1 

c = np.linspace(0, 1, 20) 

# c = 1 - Sigmaa/Sigmat 
Sigmaa = Sigmat*(1 - c) 

tol = 1e-6 

x, phi, it = sourceIteration(Sn(N, .1, .83, 1, xb=1), tol)

plt.plot(x, phi)
plt.show()

# it = np.zeros(len(Sigmaa))

# for i in range(len(Sigmaa)):

# 	sn = Sn(N, Sigmaa[i], Sigmat, q, xb=20)

# 	x, phi, it[i] = sourceIteration(sn, tol)

# plt.plot(c, it)
# plt.show()

