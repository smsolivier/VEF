#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

class Direct:
	''' initialize direct S2 solver ''' 

	def __init__(self, xe, Sigmaa, Sigmat, BCL=0, BCR=1):

		self.N = np.shape(xe)[0] - 1 # number of cell centers 
		self.Ne = np.shape(xe)[0] # number of cell edges 
		
		# boundary conditions 
		self.BCL = BCL
		self.BCR = BCR 

		self.h = np.zeros(self.N) # cell widths at cell center 
		self.xc = np.zeros(self.N) # cell centered locations 
		self.xe = xe # cell edged locations 
		self.xb = xe[-1] # edge of domain 

		# get cell centers and cell widths 
		for i in range(1, self.Ne):

			self.xc[i-1] = .5*(xe[i] + xe[i-1]) # get cell centers 
			self.h[i-1] = xe[i] - xe[i-1] # cell widths 

		# make material properties public 
		self.Sigmaa = Sigmaa 
		self.Sigmat = Sigmat 
		self.Sigmas = lambda x: Sigmat(x) - Sigmaa(x) 

		# angles for S2 
		self.mu, self.w = np.polynomial.legendre.leggauss(2) 

class DD:
	''' Direct Diamond Differenced S2 solver ''' 

	def __init__(self, xe, Sigmaa, Sigmat, BCL=0, BCR=1):

		# call direct initialization 
		Direct.__init__(self, xe, Sigmaa, Sigmat, BCL, BCR)

		# initialize matrix 
		A = np.zeros((2*self.Ne, 2*self.Ne))

		# interior cells 
		# psi_+ 
		ii = 0 # track which cell center 
		for i in range(2, 2*self.Ne, 2):

			# cell centered properties 
			Sigmati = self.Sigmat(self.xc[ii])
			Sigmasi = self.Sigmas(self.xc[ii])
			h = self.h[ii] 

			# psi_+,i-1/2 
			A[i,i-2] = -self.mu[1] + Sigmati*h/2 - Sigmasi*h/4 

			# psi_-,i-1/2 
			A[i,i-1] = -Sigmasi*h/4 

			# psi_+,i+1/2 
			A[i,i] = self.mu[1] + Sigmati*h/2 - Sigmasi*h/4 

			# psi_-,i+1/2 
			A[i,i+1] = -Sigmasi*h/4 

			ii += 1

		# psi_- 
		ii = 0 
		for i in range(1, 2*self.Ne-1, 2):

			Sigmati = self.Sigmat(self.xc[ii]) 
			Sigmasi = self.Sigmas(self.xc[ii])
			h = self.h[ii] 

			# psi_+,i-1/2 
			A[i,i-1] = -Sigmasi*h/4 

			# psi_-,i-1/2 
			A[i,i] = np.fabs(self.mu[0]) + .5*Sigmati*h - Sigmasi*h/4 

			# psi_+,i+1/2 
			A[i,i+1] = -Sigmasi*h/4 

			# psi_-,i+1/2 
			A[i,i+2] = -np.fabs(self.mu[0]) + Sigmati*h/2 - Sigmasi*h/4 

			ii += 1 

		# boundary conditions 
		# left
		if (self.BCL == 0 and self.BCR == 1): # left reflecting, right vacuum 

			# left reflecting 
			A[0,0] = 1 
			A[0,1] = -1 

			# right vacuum 
			A[-1,-1] = 1 

		elif (self.BCL == 1 and self.BCR == 1): # right/left vacuum 

			# left vacuum 
			A[0,0] = 1 

			# right vacuum 
			A[-1,-1] = 1 

		else:

			print('\n--- WARNING: BC not supported in direct.py ---\n')
			sys.exit()

		self.A = A 

	def MMS(self):
		''' setup MMS q 
			force phi = sin(pi*x/xb)
		''' 

		assert (self.BCL == 1)
		assert (self.BCR == 1)

		q = np.zeros((2, self.N))

		# loop through all angles 
		for i in range(2):

			# loop through space 
			for j in range(self.N):

				q[i,j] = self.mu[i]*np.pi/self.xb * \
					np.cos(np.pi*self.xc[j]/self.xb) + (self.Sigmat(self.xc[j]) - 
						self.Sigmas(self.xc[j]))*np.sin(np.pi*self.xc[j]/self.xb)

		# solve 
		x, phi = self.solve(q)

		return x, phi 

	def solve(self, q):
		''' solve for flux 
			q must be cell centered 
		''' 

		# make b 
		b = np.zeros(2*self.Ne) 
		ii = 0 
		for i in range(1, 2*self.Ne-1, 2):

			b[i] = q[0,ii] * self.h[ii] / 2 
			b[i+1] = q[1,ii]* self.h[ii] / 2 

			ii += 1 

		# solve for psi (order +, -)
		psi = np.linalg.solve(self.A, b)

		# store left and right psi 
		psiR = np.zeros(self.Ne)
		psiL = np.zeros(self.Ne)

		# extract left and right 
		ii = 0 
		for i in range(0, 2*self.Ne, 2):

			psiR[ii] = psi[i] 

			ii += 1 

		ii = 0 
		for i in range(1, 2*self.Ne, 2):

			psiL[ii] = psi[i] 

			ii += 1 

		# compute flux 
		phi = psiL + psiR 

		return self.xe, phi 

if __name__ == '__main__':

	from scipy.interpolate import interp1d 
	
	xb = 2

	Sigmaa = lambda x: .1 
	Sigmat = lambda x: .83 

	N = np.array([40, 80, 160])

	direct = [DD(np.linspace(0, xb, n), Sigmaa, Sigmat, 
		BCL=1, BCR=1) for n in N]

	err = np.zeros(len(N))

	phi_mms = lambda x: np.sin(np.pi*x/xb) # exact solution 

	for i in range(len(N)):

		x, phi = direct[i].MMS() 

		phif = interp1d(x, phi)

		err[i] = np.fabs(phif(xb/2) - phi_mms(xb/2))/phi_mms(xb/2)

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0])

	plt.loglog(1/N, err, '-o')
	plt.show()