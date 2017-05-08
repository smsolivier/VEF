#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from transport import * # general transport class 

from mhfem_acc import * # MHFEM acceleration solver

from directld import * # direct S2 LLDG solver 

''' Lumped Linear Discontinuous Galerkin spatial discretization of Sn 
	Inherits from transport.py 
	Includes 
		Unaccelerated
		VEF acceleration 
		Linear S2SA 
''' 

class LD(Transport):
	''' Linear Discontinuous Galerkin spatial discretization of Sn 
		Inherits functions from transport.py 
	''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call transport initialization 
		Transport.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'LD' # name of method 

		# create LD specific variables 
		# store LD left and right discontinuous points 
		self.psiL = np.zeros((self.n, self.N)) # LD left point  
		self.psiR = np.zeros((self.n, self.N)) # LD right point 

		# LD scalar fluxes 
		self.phiL = np.zeros(self.N) 
		self.phiR = np.zeros(self.N)

		# use cell centers in source iteration 
		self.x = self.xc

	def fullSweep(self, phiL, phiR):
		''' sweep left to right or right to left depending on boundary conditions ''' 

		if (self.BCL == 0 and self.BCR == 1): # left reflecting 

			self.sweepRL(phiL, phiR)
			self.sweepLR(phiL, phiR)

		elif (self.BCR == 0 and self.BCL == 1): # right reflecting 

			self.sweepLR(phiL, phiR)
			self.sweepRL(phiL, phiR)

		else: # both vacuum

			self.sweepLR(phiL, phiR)
			self.sweepRL(phiL, phiR)

	def sweep(self, phiL, phiR):
		''' unaccelerated sweep 
			sweep with given scalar flux 
		''' 

		# sweep left to right or right to left first depending on BCs 
		# get psiL, psiR (stored as object variables)
		self.fullSweep(phiL, phiR)

		# get left flux (cell centered left value) 
		phiL = self.zeroMoment(self.psiL)

		# right flux (cell centered LD right value)
		phiR = self.zeroMoment(self.psiR)

		return phiL, phiR 

	def sweepLR(self, phiL, phiR):
		''' sweep left to right (mu > 0) ''' 

		A = np.zeros((2,2)) # store psiL, psiR coefficients 
		b = np.zeros(2) # store rhs 

		# loop through positive angles 
		for i in range(int(self.n/2), self.n):

			# loop from 1:N, assume BC already set 
			for j in range(0, self.N):

				h = self.h[j] # cell width 

				# left equation 
				A[0,0] = self.mu[i] + self.Sigmat(self.xc[j])*h # psiL 
				A[0,1] = self.mu[i] # psiR 

				# right equation 
				A[1,0] = -self.mu[i] # psiL
				A[1,1] = self.mu[i] + self.Sigmat(self.xc[j])*h # psiR 

				# rhs 
				b[0] = self.Sigmas(self.xc[j])*h/2*phiL[j] + self.q[i,j]*h/2 # left 
				if (j == 0): # boundary condition 

					# default to vacuum 

					if (self.BCL == 0): # reflecting 

						b[0] += 2*self.mu[i]*self.psiL[self.mu == -self.mu[i],0]

				else: # normal sweep 

					b[0] += 2*self.mu[i]*self.psiR[i,j-1] # upwind term 

				b[1] = self.Sigmas(self.xc[j])*h/2*phiR[j] + self.q[i,j]*h/2 # right 

				ans = np.linalg.solve(A, b) # solve for psiL, psiR 

				# extract psis 
				self.psiL[i,j] = ans[0] 
				self.psiR[i,j] = ans[1] 

	def sweepRL(self, phiL, phiR):
		''' sweep right to left (mu < 0) ''' 

		A = np.zeros((2,2)) # store psiL, psiR coefficients 
		b = np.zeros(2) # store rhs 

		# loop through negative angles 
		for i in range(int(self.n/2)):

			# loop backwards from N-1:0, assume BC already set 
			for j in range(self.N-1, -1, -1):

				h = self.h[j] # cell width 

				# right equation 
				A[1,0] = -self.mu[i] # psiL
				A[1,1] = -self.mu[i] + self.Sigmat(self.xc[j])*h # psiR 

				# left equation 
				A[0,0] = -self.mu[i] + self.Sigmat(self.xc[j])*h # psiL 
				A[0,1] = self.mu[i]

				# rhs 
				b[0] = self.Sigmas(self.xc[j])*h/2*phiL[j] + self.q[i,j]*h/2 # left 
				b[1] = self.Sigmas(self.xc[j])*h/2*phiR[j] + self.q[i,j]*h/2 # right 

				if (j == self.N-1): # boundary condition 
					
					# default to vacuum 

					if (self.BCR == 0): # reflecting 

						b[1] -= 2*self.mu[i]*self.psiR[self.mu == -self.mu[i],-1]
	
				else: # normal sweep
				
					b[1] -= 2*self.mu[i]*self.psiL[i,j+1] # downwind term 

				ans = np.linalg.solve(A, b) # solve for psiL, psiR 

				# extract psis 
				self.psiL[i,j] = ans[0] 
				self.psiR[i,j] = ans[1] 

	def centPsi(self):
		''' Get cell centered psi
			psi_i = .5*(psi_i,L + psi_i,R)
		''' 

		psi = .5 * (self.psiL + self.psiR) 

		return psi 

	def edgePsi(self):
		''' get celled edge angular flux accounting for up/down winding ''' 

		psi = np.zeros((self.n, self.N+1)) # store cell edge psi 

		for i in range(self.n):

			if (self.mu[i] > 0): # positive angles 

				# set boundary values 
				if (self.BCL == 0): # reflecting 

					psi[i,0] = self.psiL[self.mu == -self.mu[i],0] 

				elif (self.BCL == 1):

					psi[i,0] = 0 

				for j in range(self.N):

					psi[i,j+1] = self.psiR[i,j] # psi_j+1/2 = psi_i,R  

			else: # negative angles 

				# set boundary values 
				if (self.BCR == 0): # reflecting 

					psi[i,-1] = self.psiR[self.mu == -self.mu[i],-1]

				elif (self.BCR == 1): # vacuum 

					psi[i,-1] = 0 

				for j in range(self.N-1, -1, -1):

					psi[i,j] = self.psiL[i,j] # psi_j-1/2 = psi_i,L

		self.psi = psi

		return psi 

	def sourceIteration(self, tol, maxIter=200):
		''' LD source iteration 
			converges phiL and phiR separately 
			Inputs:
				tol: convergence tolerance 
				maxIter: maximum number of iterations 
		''' 

		it = 0 # store number of iterations 

		tt = Timer.timer() # start timer 

		# lambda function to compute convergence criterion 
		conv_f = lambda new, old: np.linalg.norm(new - old, 2)/np.linalg.norm(new, 2)

		self.diff = [] # store || phi^ell+1 - phi^ell || 

		self.phiConv = [] # store convergence of flux 
		self.eddConv = [] # store convergence of eddington 

		edd = np.zeros(self.N) # cell centered eddington 

		while (True):

			# check if max reached 
			if (it == maxIter):

				print('\n--- WARNING: maximum number of source iterations reached ---\n')
				break 

			# store old flux 
			phiL_old = np.copy(self.phiL)
			phiR_old = np.copy(self.phiR)

			# store old eddington 
			edd_old = np.copy(edd)

			# sweep to update flux 
			self.phiL, self.phiR = self.sweep(phiL_old, phiR_old)

			# pointwise convergence 
			# returns array of convergence values as each spatial location 
			convL_point = self.pointConvergence(self.phiL, phiL_old)
			convR_point = self.pointConvergence(self.phiR, phiR_old) 

			# get max value 
			convL = np.max(convL_point)
			convR = np.max(convR_point)

			# store eddington convergence 
			edd = self.getEddington(.5*(self.psiL + self.psiR))
			self.eddConv.append(np.max(self.pointConvergence(edd, edd_old)))

			# store average of left and right 
			self.phiConv.append((convL + convR)/2) 

			# spectral radius 
			self.diff.append(np.linalg.norm(.5*(self.phiL + self.phiR) - 
				.5*(phiL_old + phiR_old), 1))

			# check for convergence 
			if (convL < tol and convR < tol): 

				break # exit loop if converged 

			# update iteration count 
			it += 1 

			# print to terminal: number of iterations, distance until tolerance is reached 
			print('{} {}'.format(it, .5*(convL + convR)/tol), end='\r')

		print('Number of Iterations =', it, end=', ')
		tt.stop() # end timer 

		# combine phiL and phiR 
		phi = .5*(self.phiL + self.phiR) # average of left and right 

		return self.x, phi, it 

class Eddington(LD):
	''' Eddington accelerated LD ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1, OPT=1, GAUSS=1):
		''' OPT: controls how LD left and right fluxes are recovered
				0: use cell centers
				1: maintain slopes by using the cell edges from MHFEM 
				2: van Leer slope reconstruction on centers only 
			GAUSS: use gauss quad for <mu^2> in MHFEM 
				0: computes <mu^2> from edge and center psi 
				1: uses gauss quad to compute centers for rational polynomial <mu^2> 
					edges from edge psi 
		''' 

		# call LD initialization 
		LD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'LD Edd' + str(OPT) + str(GAUSS) # name of method 

		self.OPT = OPT # slope recovery method 

		self.GAUSS = GAUSS # use gauss quad in eddington creation 

		# use cell centers in source iteration 
		self.x = self.xc 

		# create MHFEM object, return edge and center values 
		self.mhfem = MHFEM(self.xe, self.Sigmaa, self.Sigmat, 
			self.BCL, self.BCR, CENT=2)

	def ldRecovery(self, phi, OPT=1):
		''' Recover LD left and right values 
			OPT: determine which recovery scheme to use 
				0: use edges (calls useHalf)
				1: maintain average and slope (calls maintainSlopes)
				2: van Leer on centers (calls reconstructSlopes)
		'''

		if (OPT == 0):

			phiL, phiR = self.useHalf(phi) # use edges from MHFEM 

		elif (OPT == 1):

			phiL, phiR = self.maintainSlopes(phi) # recover slopes from edges

		elif (OPT == 2):

			phiL, phiR = self.reconstructSlopes(phi) # recover slopes from centers only 

		return phiL, phiR 

	def maintainSlopes(self, phi):
		''' maintain cell center value and slopes 
			Reconstructs from cell edges 
		''' 

		phiL = np.zeros(self.N) # left flux 
		phiR = np.zeros(self.N) # right flux 

		# separate edges and centers 
		phiEdge = self.mhfem.getEdges(phi) # get edges from MHFEM object utility 
		phiCent = self.mhfem.getCenters(phi) # get centers from MHFEM object utility 

		for i in range(self.N):

			# phi_i+1/2 - phi_i-1/2 
			diff = .5*(phiEdge[i+1] - phiEdge[i]) 

			# recover left and right 
			phiL[i] = phiCent[i] - diff 

			phiR[i] = phiCent[i] + diff

		return phiL, phiR 

	def useHalf(self, phi):
		''' use phi_iL = phi_i-1/2, phi_iR = phi_i+1/2 ''' 

		phiL = np.zeros(self.N)
		phiR = np.zeros(self.N) 

		phiEdge = self.mhfem.getEdges(phi) # get edges from MHFEM object utility 

		for i in range(self.N):

			phiL[i] = phiEdge[i] 
			phiR[i] = phiEdge[i+1] 

		return phiL, phiR 

	def reconstructSlopes(self, phi):
		''' Reconstruct phiL, phiR from MHFEM cell centers only 
			Inputs:
				phi: MHFEM scalar flux (edges and centers) 
			''' 

		omega = 0 # weighting factor for slope reconstruction 

		# extract cell centers 
		phiC = self.mhfem.getCenters(phi)

		# compute delta 
		delta = np.zeros(self.Ne) # cell edged differences 

		for i in range(1, self.N):

			delta[i] = phiC[i] - phiC[i-1]

		delta[0] = delta[1] 
		delta[-1] = delta[-2] 

		# get slope limiter 
		xi = self.vanLeer(delta, omega)

		# compute slopes 
		slope = np.zeros(self.N) # cell centered 

		for i in range(self.N):

			slope[i] = .5*(1 + omega)*delta[i] + .5*(1-omega)*delta[i+1] 

		# apply limiter 
		slope *= xi 

		# reconstuct LD left and right 
		phiL = np.zeros(self.N)
		phiR = np.zeros(self.N)

		for i in range(self.N):

			phiL[i] = phiC[i] - slope[i] / 2
			phiR[i] = phiC[i] + slope[i] / 2

		return phiL, phiR

	def vanLeer(self, delta, omega):
		''' Generate van leer slope limiter 
			Inputs:
				delta: cell edged center differences 
				omega: left/right difference weighting parameter
		''' 

		xi = np.zeros(self.N) # self centered limiter 

		r = np.zeros(self.N) # centered ratio of edge deltas 

		beta = 1 

		for i in range(self.N):

			if (delta[i+1] != 0):

				r[i] = delta[i]/delta[i+1] 

		for i in range(self.N):

			if r[i] < 0:

				xi[i] = 0 

			else:

				xiR = 2*beta/(1 - omega + (1+omega)*r[i])
				xi[i] = min(2*r[i]/(1+r[i]), xiR)

		return xi

	def makeEddingtonGauss(self):
		''' Create edge and center array of eddington factor 
			Edges from upwinded edge psi 
			Centers from Gauss Quadrature second order approximation of integrating 
				linear eddington 
		''' 

		# get edge eddington from upwinded values 
		psiEdge = self.edgePsi() # get edge values 
		mu2_edge = self.getEddington(psiEdge) # edge eddington 

		# evaluate centers with gauss quad 
		phiL = self.zeroMoment(self.psiL) # left flux 
		phiR = self.zeroMoment(self.psiR) # right flux 

		# integrate mu^2 psi 
		muPsiL = np.zeros(self.N) # left 
		muPsiR = np.zeros(self.N) # right 
		for i in range(self.n):

			muPsiL += self.mu[i]**2 * self.psiL[i,:] * self.w[i] 
			muPsiR += self.mu[i]**2 * self.psiR[i,:] * self.w[i] 

		mu2_cent = np.zeros(self.N) # cell centered eddington

		# MHFEM basis functions 
		Bli = lambda x, i: (self.xe[i+1] - x)/self.h[i] 
		Bri = lambda x, i: (x - self.xe[i])/self.h[i] 

		# compute interior coefficients 
		for i in range(self.N):

			# convert from (-1, 1) --> (x_i-1/2, x_i+1/2)
			xlg = self.xc[i] - self.h[i]/2 / np.sqrt(3) 
			xrg = self.xc[i] + self.h[i]/2 / np.sqrt(3) 

			xg = np.array([xlg, xrg]) # combine left and right points 

			# compute center with 2 order gauss quad 
			for j in range(len(xg)):

				mu2_cent[i] += (Bli(xg[j], i) * muPsiL[i] + Bri(xg[j], i) * muPsiR[i])/(
					phiL[i] * Bli(xg[j], i) + phiR[i] * Bri(xg[j], i)) / 2

		# concatenate into one array 
		mu2 = np.zeros(2*self.Ne - 1) # store centers and edges 
		mu2[0] = mu2_edge[0] # set left boundary 
		ii = 1 
		for i in range(self.N):

			mu2[ii] = mu2_cent[i] 
			mu2[ii+1] = mu2_edge[i+1] 

			ii += 2 

		return mu2 

	def makeEddingtonConst(self):
		''' Create edge and center array of eddington factor 
			Edges from upwinded edge psi 
			Centers from average of psiL and psiR
		''' 

		psiEdge = self.edgePsi() # get edge psi 
		psiCent = self.centPsi() # get center psi 

		mu2_edge = self.getEddington(psiEdge) # edge eddington factor 
		mu2_cent = self.getEddington(psiCent) # center eddington factor 

		# concatenate into one array 
		mu2 = np.zeros(2*self.Ne - 1) # store centers and edges 
		mu2[0] = mu2_edge[0] # set left boundary 
		ii = 1 
		for i in range(self.N):

			mu2[ii] = mu2_cent[i] 
			mu2[ii+1] = mu2_edge[i+1] 

			ii += 2 

		return mu2 

	def sweep(self, phiL, phiR):
		''' one source iteration sweep 
			Overwrites sweep in LD class to include eddington acceleration 
		''' 

		self.fullSweep(phiL, phiR) # transport sweep, BC dependent ordering 

		# make SN flux public 
		self.phi_SN = self.zeroMoment(self.centPsi())

		# create eddington for MHFEM 
		if (self.GAUSS == 0):

			self.mu2 = self.makeEddingtonConst()

		elif(self.GAUSS == 1):

			self.mu2 = self.makeEddingtonGauss()

		else:

			print('\n --- FATAL ERROR: LD Eddington GAUSS not defined properly ---\n')
			sys.exit()

		psiEdge = self.edgePsi()

		# create boundary eddington factor 
		top = 0 
		for i in range(self.n):

			top += np.fabs(self.mu[i])*psiEdge[i,:] * self.w[i] 

		B = top/self.zeroMoment(psiEdge) 

		# discretize MHFEM
		self.mhfem.discretize(self.mu2, B)

		# solve for phi, get edges and centers 
		x, phi = self.mhfem.solve(self.zeroMoment(self.q)/2, 
			self.firstMoment(self.q)/2)

		# get LD left and right fluxes from MHFEM flux 
		phiL, phiR = self.ldRecovery(phi, OPT=self.OPT)

		return phiL, phiR # return accelerated flux 

class S2SA(LD):
	''' LLDG Linear S2SA solver ''' 

	def __init__(self, xe, n, Sigmaa, Sigmat, q, BCL=0, BCR=1):

		# call LD initialization 
		LD.__init__(self, xe, n, Sigmaa, Sigmat, q, BCL, BCR)

		self.name = 'LD S2SA'

		# create direct solve object 
		self.direct = Direct(self.xe, self.Sigmaa, self.Sigmat, BCL, BCR)

	def sweep(self, phiL, phiR):
		''' S2SA sweep ''' 

		# do normal sweep 
		self.fullSweep(phiL, phiR)

		# compute scalar flux 
		phihalfL = self.zeroMoment(self.psiL)
		phihalfR = self.zeroMoment(self.psiR)

		# s2sa source 
		qL = self.Sigmas(self.xc)*(phihalfL - phiL)
		qR = self.Sigmas(self.xc)*(phihalfR - phiR)

		# solve for update factor 
		x, fL, fR = self.direct.solve(qL, qR)

		return phihalfL + fL, phihalfR + fR

if __name__ == '__main__':

	N = 20 # number of spatial cells 
	n = 8 # number of discrete ordinates 
	xb = 2 # domain boundary 
	x = np.linspace(0, xb, N+1) # cell edge locations 

	# material properties 
	Sigmaa = lambda x: .1
	Sigmat = lambda x: 1
	q = lambda x, mu: 1

	# iterative convergence tolerance 
	tol = 1e-6

	# create solver objects 
	ld = LD(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)
	# ld.setMMS()
	ed = Eddington(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1, OPT=2)
	# ed.setMMS()
	s2 = S2SA(x, n, Sigmaa, Sigmat, q, BCL=0, BCR=1)
	# s2.setMMS() 

	# run source iteration 
	x, phi, it = ld.sourceIteration(tol)

	xe, phie, ite = ed.sourceIteration(tol)

	x2, phi2, it2 = s2.sourceIteration(tol)

	# plot results 
	plt.figure()
	plt.plot(x, phi, label='LD')
	plt.plot(xe, phie, label='LD Edd')
	plt.plot(x2, phi2, label='S2SA')
	plt.legend(loc='best')
	plt.show()