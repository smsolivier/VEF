#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import mhfem_acc as mh

import Timer 

class DD:
	''' Diamond (Crank Nicolson) differenced transport 
		mu dpsi/dx + sigmat psi = sigmas/2 phi + Q/2 
		parent class containing:
			left to right sweeping 
			right to left sweeping 
			gauss legendre psi integrator 
			source iteration 
			<mu^2> generator 
	'''  

	def __init__(self, x, n, Sigmaa, Sigmat, q):
		''' Inputs:
				x: locations of cell edges 
				n: number of discrete ordinates 
				Sigmaa: absorption XS function  
				Sigmat: total XS function  
				q: source function
		''' 

		self.N = np.shape(x)[0] # number of cell edges 
		self.n = n # number of discrete ordinates 

		self.x = x # cell edge locations 

		assert(n%2 == 0) # assert n is even 

		# make material properties public 
		self.Sigmaa = Sigmaa # absorption XS function 
		self.Sigmat = Sigmat # total XS function 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x) # scattering XS function 
		self.q = q # fixed source array, cell edged 

		# store all psis at each cell edge 
		self.psiL = np.zeros((n, self.N)) # mu < 0  
		self.psiR = np.zeros((n, self.N)) # mu > 0 

		# angular flux for all mu_n and all x locations 
		self.psi = np.zeros((n, self.N)) 

		self.phi = np.zeros(self.N) # store flux 

		# generate mu's, mu arranged negative to positive 
		self.mu, self.w = np.polynomial.legendre.leggauss(n)

	def sweep(self, phi):
		''' unaccelerated sweep with reflecting left and vacuum right boundary ''' 

		# sweep right to left, default to vacuum 
		self.sweepRL(phi)

		# set left boundary 
		for i in range(int(self.n/2), self.n):

			self.psi[i,0] = self.psi[self.mu == -self.mu[i],0]

		self.sweepLR(phi) 

		return self.integratePsi()

	def sweepRL(self, phi):
		''' sweep right to left (mu < 0) ''' 

		# loop through negative angles  
		for i in range(int(self.n/2)):

			# spatial loop from right to left 
			for j in range(self.N-2, -1, -1):

				midpoint = (self.x[j] + self.x[j+1])/2 # location of cell center 
				h = self.x[j+1] - self.x[j] # cell width 

				# rhs 
				b = self.Sigmas(midpoint)/4*(phi[j] + phi[j+1]) + .25*(self.q[j] + self.q[j+1])

				self.psi[i,j] = b*h - (.5*self.Sigmat(midpoint)*h - 
					np.fabs(self.mu[i]))*self.psi[i,j+1]

				self.psi[i,j] /= .5*self.Sigmat(midpoint)*h + np.fabs(self.mu[i])

	def sweepLR(self, phi):
		''' sweep left to right (mu > 0) ''' 

		# loop through positive angles 
		for i in range(int(self.n/2), self.n):

			# spatial loop from left to right 
			for j in range(1, self.N):

				midpoint = (self.x[j] + self.x[j-1])/2 # cell center 
				h = self.x[j] - self.x[j-1] # cell width 

				# rhs 
				b = self.Sigmas(midpoint)/4*(phi[j] + phi[j-1]) + .25*(self.q[j] + self.q[j-1])

				self.psi[i,j] = b*h - (.5*self.Sigmat(midpoint)*h - 
					self.mu[i])*self.psi[i,j-1]

				self.psi[i,j] /= .5*self.Sigmat(midpoint)*h + self.mu[i] 

	def integratePsi(self):
		''' use Gauss quadrature to integrate psi ''' 

		phi = np.zeros(self.N) # store flux 

		# loop through angles 
		for i in range(self.n):

			phi += self.psi[i,:] * self.w[i] 

		return phi 

	def getEddington(self):
		''' use Gauss quadrature to integrate mu**2 psi ''' 

		top = 0 # int mu**2 psi dmu 
		for i in range(self.n): # loop through angles 

			top += self.mu[i]**2 * self.psi[i,:] * self.w[i]

		mu2 = top/self.integratePsi()

		return mu2 

	def sourceIteration(self, tol, PLOT=False):
		''' lag RHS of transport equation and iterate until flux converges ''' 

		it = 0 # store number of iterations 
		phi = np.zeros(self.N) # store flux at each spatial location  
		edd = np.zeros(self.N) # store eddington factor 

		self.phiCon = [] 
		self.eddCon = [] 

		tt = Timer.timer() # time source iteration convergence 

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

		print('Number of iterations =', it, end=', ') 
		tt.stop()

		if (PLOT):
			for i in range(int(self.n/2)):
				plt.plot(self.x, self.psiL[i,:], label=str(self.muL[i]))
				plt.plot(self.x, self.psiR[i,:], label=str(self.muR[i]))

			plt.legend(loc='best')
			plt.show()

		# return spatial locations, flux and number of iterations 
		return self.x, phi, it 

class Eddington(DD):
	''' Eddington Acceleration ''' 

	def sweep(self, phi):

		# transport sweep 
		# defaults to vacuum 
		self.sweepRL(phi) # sweep right to left 

		# set left boundary 
		for i in range(int(self.n/2), self.n):

			self.psi[i,0] = self.psi[self.mu == -self.mu[i], 0]

		# sweep left to right 
		self.sweepLR(phi)

		# get eddington factor 
		mu2 = self.getEddington()

		# generate boundary eddington for transport consistency 
		top = 0 
		for i in range(self.n):

			top += np.fabs(self.mu[i])*self.psi[i,:] * self.w[i]

		B = top/self.integratePsi()*2

		# create MHFEM object 
		sol = mh.MHFEM(self.x, mu2, self.Sigmaa, self.Sigmat, B, BCL=0, BCR=1)

		# solve for drift diffusion flux 
		x, phi = sol.solve(self.q)

		return phi # return MHFEM flux 

class DSA(DD):
	''' Diffusion Synthetic Acceleration ''' 

	# override DD initialization 
	def __init__(self, x, n, Sigmaa, Sigmat, q):

		# call DD initialization 
		DD.__init__(self, x, n, Sigmaa, Sigmat, q)

		# create fem object on top of standard Transport initialization 
		self.fem = mh.MHFEM(self.x, np.ones(self.N)/3, self.Sigmaa, 
			self.Sigmat, B=np.ones(self.N), BCL=0, BCR=1)

	def sweep(self, phi):

		# transport sweep 
		# defaults to vacuum 
		self.sweepRL(phi) # sweep right to left 

		# set left boundary 
		for i in range(int(self.n/2), self.n):

			self.psi[i,0] = self.psi[self.mu == -self.mu[i], 0]

		# sweep left to right 
		self.sweepLR(phi)

		# compute phi^(l+1/2)
		phihalf = self.integratePsi()

		# DSA step 
		x, f = self.fem.solve(self.Sigmas(self.x)*(phihalf - phi))

		# return updated flux 
		return phihalf + f 

if __name__ == '__main__':

	N = 50
	n = 8
	xb = 2 
	x = np.linspace(0, xb, N)

	Sigmaa = lambda x: .1 
	Sigmat = lambda x: .83 

	q = np.ones(N)

	ed = Eddington(x, n, Sigmaa, Sigmat, q)
	dd = DD(x, n, Sigmaa, Sigmat, q)
	dsa = DSA(x, n, Sigmaa, Sigmat, q)

	x, phi, it = dd.sourceIteration(1e-6)
	xe, phie, ite = ed.sourceIteration(1e-6)
	xd, phid, itd = dsa.sourceIteration(1e-6)

	plt.plot(x, phi, label='DD')
	plt.plot(xe, phie, label='Edd')
	plt.plot(xd, phid, label='DSA')
	plt.legend(loc='best')
	plt.show()