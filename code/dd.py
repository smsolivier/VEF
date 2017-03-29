#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from transport import * # general transport class 

from mhfem_acc import * 

import direct as Direct # direct S2 solver 

''' Diamond differenced Sn solver 
	Inherits from transport.py 
	Includes 
		unaccelerated 
		eddington accelerated 
		S2SA 
		inconsistent DSA
''' 

class DD(Transport):
	''' Diamond Difference spatial discretization of Sn 
		Inherits functions from transport.py 
	''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call transport initialization 
		Transport.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'DD' # name of method 

	def fullSweep(self, phi):
		''' set sweep order based on boundary conditions ''' 

		if (self.BCL == 0 and self.BCR == 1): # left reflecting 

			# default vacuum 
			self.sweepRL(phi)

			# set left reflecting BC 
			for i in range(int(self.n/2), self.n):

				self.psi[i,0] = self.psi[self.mu == -self.mu[i],0]

			# sweep back 
			self.sweepLR(phi) 

		elif (self.BCR == 0 and self.BCL == 1): # right reflecting 

			# default vacuum 
			self.sweepLR(phi)

			# set left reflecting BC 
			for i in range(int(self.n/2)):

				self.psi[i,-1] = self.psi[self.mu == -self.mu[i],-1]

			# sweep back 
			self.sweepRL(phi) 

		else:

			# default vacuum 
			self.sweepLR(phi)

			# default vacuum 
			self.sweepRL(phi)

	def sweep(self, phi):
		''' unaccelerated sweep with reflecting left and vacuum right boundary ''' 

		phi = self.getCenter(phi) # convert cell edges to cell centers 

		self.fullSweep(phi)

		return self.zeroMoment(self.psi) # return flux 

	def sweepRL(self, phi):
		''' sweep right to left (mu < 0) ''' 

		# loop through negative angles  
		for i in range(int(self.n/2)):

			# spatial loop from right to left 
			for j in range(self.Ne-2, -1, -1):

				midpoint = self.xc[j] # location of cell center 
				h = self.h[j] # cell width 

				# rhs 
				b = self.Sigmas(midpoint)/2*phi[j] + .5*self.q[i,j]

				self.psi[i,j] = b*h - (.5*self.Sigmat(midpoint)*h - 
					np.fabs(self.mu[i]))*self.psi[i,j+1]

				self.psi[i,j] /= .5*self.Sigmat(midpoint)*h + np.fabs(self.mu[i])

	def sweepLR(self, phi):
		''' sweep left to right (mu > 0) ''' 

		# loop through positive angles 
		for i in range(int(self.n/2), self.n):

			# spatial loop from left to right 
			for j in range(1, self.Ne):

				h = self.h[j-1] # cell width 
				midpoint = self.xc[j-1] # cell center 

				# rhs 
				b = self.Sigmas(midpoint)/2*phi[j-1] + .5*self.q[i,j-1]

				self.psi[i,j] = b*h - (.5*self.Sigmat(midpoint)*h - 
					self.mu[i])*self.psi[i,j-1]

				self.psi[i,j] /= .5*self.Sigmat(midpoint)*h + self.mu[i] 

	def getCenter(self, phi):
		''' convert cell edged flux to cell center by taking 
			average of cell edge values 
		''' 

		phiA = np.zeros(self.N) # cell centered flux 

		for i in range(1, self.Ne):

			phiA[i-1] = .5*(phi[i] + phi[i-1])

		return phiA 

class Eddington(DD):
	''' Eddington Acceleration ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):
		''' CENT: controls whether to get edges or centers from MHFEM
				0: edges 
				1: centers 
		''' 

		# call DD initialization 
		DD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'DD Edd' # name of method 

		# create MHFEM solver 
		self.mhfem = MHFEM(self.xe, self.Sigmaa, self.Sigmat, 
			self.BCL, self.BCR, CENT=1)

		# cell edged flux 
		self.phi = np.zeros(self.N)

		# use cell centers in SI 
		self.x = self.xc

	def sweep(self, phi):

		# sweep with BCs 
		self.fullSweep(phi)

		# cell centered psi 
		psi = np.zeros((self.n, self.N))

		for i in range(self.n):

			psi[i,:] = self.getCenter(self.psi[i,:])

		# get eddington factor 
		mu2 = self.getEddington(psi)

		# make SN phi public 
		self.phi_SN = self.zeroMoment(psi)

		# generate boundary eddington for transport consistency 
		top = 0 
		for i in range(self.n):

			top += np.fabs(self.mu[i]) * self.psi[i,:] * self.w[i]

		B = top/self.zeroMoment(self.psi)

		# discretize MHFEM with mu2 and B 
		self.mhfem.discretize(mu2, B)

		# solve for drift diffusion flux 
		x, phi = self.mhfem.solve(self.zeroMoment(self.q)/2, 
			self.firstMoment(self.q)/2)

		return phi # return MHFEM flux 

class DSA(DD):
	''' Inconsistent Diffusion Synthetic Acceleration ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call DD initialization 
		DD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'DD DSA' # name of method 

		# create MHFEM object 
		self.mhfem = MHFEM(self.xe, self.Sigmaa, self.Sigmat, BCL, BCR, CENT=0)

		# discretize MHFEM 
		self.mhfem.discretize(np.ones(self.Ne)/3, np.ones(self.Ne)/2)

	def sweep(self, phi):

		phiC = self.getCenter(phi)

		self.fullSweep(phiC)

		# compute phi^(l+1/2)
		phihalf = self.zeroMoment(self.psi)

		# DSA step 
		x, f = self.mhfem.solve(self.Sigmas(self.xe)*(phihalf - phi), 
			np.zeros(self.Ne))

		# return updated flux 
		return phihalf + f 

class S2SA(DD):
	''' S2 Synthetic Acceleration ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call DD initialization 
		DD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'DD S2SA' # name of method 

		# create DD direct solver 
		self.direct = Direct.DD(self.xe, self.Sigmaa, self.Sigmat, BCL, BCR)

		# use cell centered flux 
		self.phi = np.zeros(self.N)
		self.x = self.xc

	def sweep(self, phi):

		# sweep 
		self.fullSweep(phi)

		# edge flux half step 
		phihalf = self.zeroMoment(self.psi) 

		# switch to centers
		phihalf = self.getCenter(phihalf)

		# create S2SA source 
		q = np.zeros((2, self.N))
		q[0,:] = self.Sigmas(self.xc)*(phihalf - phi)
		q[1,:] = q[0,:]

		# solve for update factor 
		x, f = self.direct.solve(q)

		# switch to centers
		f = self.getCenter(f)

		return phihalf + f 

if __name__ == '__main__':

	N = 50
	n = 8
	xb = 2 
	x = np.linspace(0, xb, N)

	eps = 1

	Sigmaa = lambda x: .1 * eps
	Sigmat = lambda x: .83 / eps 

	qf = lambda x: (x > xb/2)

	q = np.ones((n,N-1)) * eps

	for i in range(n):

		q[i,:] = qf(np.linspace(xb/N/2, xb - xb/N/2, N-1)) 

	BCL = 0

	tol = 1e-8

	dd = DD(x, n, Sigmaa, Sigmat, q, BCL=BCL, BCR=1)
	# dd.setMMS()
	ed = Eddington(x, n, Sigmaa, Sigmat, q, BCL=BCL, BCR=1)
	# ed.setMMS()
	# dsa = DSA(x, n, Sigmaa, Sigmat, q, BCL=BCL, BCR=1)
	# s2sa = S2SA(x, n, Sigmaa, Sigmat, q, BCL=BCL, BCR=1)

	x, phi, it = dd.sourceIteration(tol, maxIter=1000, PLOT=None)
	xe, phie, ite = ed.sourceIteration(tol, PLOT=None)
	# xd, phid, itd = dsa.sourceIteration(tol)
	# xs, phis, its = s2sa.sourceIteration(tol, PLOT=None)

	plt.plot(x, phi, label=dd.name)
	plt.plot(xe, phie, label=ed.name)
	# plt.plot(xd, phid, label=dsa.name)
	# plt.plot(xs, phis, label=s2sa.name)
	plt.legend(loc='best')
	plt.show()