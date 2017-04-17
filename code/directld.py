#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from transport import * 

class Direct(Transport):
	''' Linear Discontinuous Galerkin direct solver ''' 

	def __init__(self, xe, Sigmaa, Sigmat, BCL=0, BCR=1):

		# call transport init 
		Transport.__init__(self, xe, 2, Sigmaa, Sigmat, lambda x, mu: 1, BCL, BCR)

		self.name = 'DirectLD'

		self.psiL = np.zeros(self.N)
		self.psiR = np.zeros(self.N)

		# mu[0] = -1/sqrt(3), mu[1] = 1/sqrt(3) 

		self.buildMatrix()

	def buildMatrix(self):

		A = np.zeros((self.N*4, self.N*4)) 

		ii = 0 # store cell center
		for i in range(0, self.N*4, 4):

			# cell centered cross sections 
			Sigmati = self.Sigmat(self.xc[ii])
			Sigmasi = self.Sigmas(self.xc[ii])
			hi = self.h[ii] # cell i width 

			# --- psi_+,i,L equation ---
			# psi_+,i,L
			A[i,i] = self.mu[1] + Sigmati*hi - Sigmasi*hi/2 

			# psi_+,i,R
			A[i,i+1] = self.mu[1] 

			# psi_-,i,L 
			A[i,i+2] = -Sigmasi*hi/2 

			# psi_+,i-1,R 
			# boundary condition 
			if (i != 0):

				A[i,i-3] = -2*self.mu[1]

			else:

				# default to vacuum (leave as zero)

				if (self.BCL == 0): # reflecting 

					# reflecting: set psi_+,i-1,R = psi_-,1,R
					A[i,i+3] += -2*self.mu[1] 

			# --- psi_+,i,R equation --- 
			i += 1 # update matrix location 

			# psi_+,i,L 
			A[i,i-1] = -self.mu[1] 

			# psi_+,i,R 
			A[i,i] = self.mu[1] + Sigmati*hi - Sigmasi*hi/2 

			# psi_-,i,R 
			A[i,i+2] = -Sigmasi*hi/2 

			# --- psi_-,i,L equation --- 
			i += 1 # update matrix location 

			# psi_+,i,L 
			A[i,i-2] = -Sigmasi*hi/2 

			# psi_-,i,L 
			A[i,i] = -self.mu[0] + Sigmati*hi - Sigmasi*hi/2 

			# psi_-,i,R 
			A[i,i+1] = self.mu[0]

			# --- psi_-,i,R equation --- 
			i += 1 # update matrix location 

			# psi_+,i,R 
			A[i,i-2] = -Sigmasi*hi/2 

			# psi_-,i,L 
			A[i,i-1] = -self.mu[0] 

			# psi_-,i,R 
			A[i,i] = -self.mu[0] + Sigmati*hi - Sigmasi*hi/2 

			if (i != self.N*4-1):

				# psi_-,i+1,L
				A[i,i+3] = 2*self.mu[0] 

			else:

				# default to vacuum 

				if (self.BCR == 0): # reflecting 

					# psi_+,N,L
					A[i,i-3] += 2*self.mu[0] 

			ii += 1 # update cell center

		self.A = A 

	def solve(self, qL, qR):

		Q = np.zeros(self.N*4) 

		ii = 0 # store cell center 
		for i in range(0, self.N*4, 4):

			h2 = self.h[ii]/2

			# positive angled Q 
			Q[i] = h2 * qL[ii] 

			Q[i+1] = h2 * qR[ii] 

			# negative angled Q 
			Q[i+2] = h2 * qL[ii] 

			Q[i+3] = h2 * qR[ii]

			ii += 1 # update cell center location 

		# solve 
		ans = np.linalg.solve(self.A, Q)

		psiPL = np.zeros(self.N) # psi_+,L
		psiPR = np.zeros(self.N) # psi_+,R
		psiML = np.zeros(self.N) # psi_-,L
		psiMR = np.zeros(self.N) # psi_-,R 

		# extract psi's 
		ii = 0 
		for i in range(0, 4*self.N, 4):

			psiPL[ii] = ans[i] 
			psiPR[ii] = ans[i+1] 
			psiML[ii] = ans[i+2] 
			psiMR[ii] = ans[i+3] 

			ii += 1 

		# compute psi_+ and psi_- 
		psiP = .5 * (psiPL + psiPR)
		psiM = .5 * (psiML + psiMR) 

		# compute scalar flux 
		phi = psiP + psiM  

		return self.xc, psiPL + psiML, psiPR + psiMR 

		# return self.xc, 2*phi

if __name__ == '__main__':
	from exactDiff import * 
	import ld as LD 

	N = 100
	xb = 50
	x = np.linspace(0, xb, N+1)

	eps = .05

	Sigmaa = lambda x: eps
	Sigmat = lambda x: 1/eps

	qL = np.ones(N)*eps
	qR = np.zeros(N)*eps

	di = Direct(x, Sigmaa, Sigmat, BCL=0, BCR=1)
	# di.setMMS()

	si = LD.Eddington(x, 2, Sigmaa, Sigmat, lambda x, mu: eps, BCL=0, BCR=1)

	xc, phiL, phiR = di.solve(qL, qR)
	# xc, phi = di.solve(di.q[0,:], di.q[0,:])

	xsi, phisi, itsi = si.sourceIteration(1e-10)

	phi_diff = exactDiff(Sigmaa(0), Sigmat(0), eps, xb, BCL=0)

	plt.plot(xc, phiL + phiR, label='Direct')
	plt.plot(xsi, phisi, label='SI')
	# plt.plot(xc, phi_diff(xc), label='Diff')
	plt.legend(loc='best')
	plt.show()