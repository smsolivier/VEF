#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fem import * 

class Transport:
	''' parent class containing:
			left to right sweeping 
			right to left sweeping 
			source iteration 
		must supply the sweep(phi) function in the inherited class for 
			SI to work 
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
		self.fem = FEM(self.N, self.Sigmaa, self.Sigmat, self.xb)

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
		x, f = self.fem.solve(self.Sigmas*(phihalf - phi)*self.h)

		# compute new flux 
		return phihalf + f 

N = 1000
Sigmat = 1 
q = 1 

c = np.linspace(0, .92, 20) 

# c = 1 - Sigmaa/Sigmat 
Sigmaa = Sigmat*(1 - c) 

# tol = 1e-6 

# x, phi, it = sourceIteration(DSA(N, .1, .83, 1, xb=1), tol)

# plt.plot(x, phi)
# plt.show()

# sn = Sn(N, Sigmaa[10], Sigmat, q, xb=1)
# dsa = DSA(N, Sigmaa[10], Sigmat, q, xb=1)
# x, phi, it = sn.sourceIteration(1e-6)
# xdsa, phidsa, itdsa = dsa.sourceIteration(1e-6)

# plt.plot(x, phi)
# plt.plot(xdsa, phidsa)
# plt.show()

it = np.zeros(len(Sigmaa))
itdsa = np.zeros(len(Sigmaa))
tol = 1e-3

for i in range(len(Sigmaa)):

	# solve Sn
	sn = Sn(N, Sigmaa[i], Sigmat, q, xb=20)
	x, phi, it[i] = sn.sourceIteration(tol)

	# solve DSA 
	dsa = DSA(N, Sigmaa[i], Sigmat, q, xb=20)
	xdsa, phidsa, itdsa[i] = dsa.sourceIteration(tol)

plt.plot(c, it, '-o', label='S2')
plt.plot(c, itdsa, '-o', label='DSA')
plt.legend(loc='best')
plt.xlabel(r'$\frac{\Sigma_s}{\Sigma_t}$')
plt.ylabel('Number of Iterations')
plt.show()

