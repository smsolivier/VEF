#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from TDMA import * 

''' Regular finite element diffusion solver ''' 

class FEM:

	def __init__(self, xe, mu2, Sigmaa, Sigmat, BCL=0, BCR=2):
		''' Solves diffusion equation with standard finite element 
			only reflecting left and marshak right BC's supported 
			Inputs:
				xe: array of cell edges 
				mu2: array of <mu^2> values at cell edges (will be interpolated)
				Sigmaa: absorption XS function 
				Sigmat: total XS function 
				BCL: left boundary 
					0: reflecting 
					1: marshak 
				BCR: right boundary 
					0: reflecting 
					1: marshak 
		''' 

		N = np.shape(xe)[0] # number of cell edges 

		h = np.zeros(N-1) # store cell widths 
		for i in range(1, N):

			h[i-1] = xe[i] - xe[i-1] # distance between cell edges 

		# build left coefficient 
		aw = np.zeros(N) 
		for i in range(1, N):

			# evaluate cross sections at cell centers 
			SigmatL = Sigmat(xe[i] - h[i-1]/2)
			SigmaaL = Sigmaa(xe[i] - h[i-1]/2)

			aw[i] = -1/(SigmatL*h[i-1])*mu2[i-1] + 1/6*SigmaaL*h[i-1]

		# build diagonal 
		ap = np.zeros(N)
		for i in range(1, N-1):

			# evaluate cross sections at cell centers 
			SigmatL = Sigmat(xe[i] - h[i-1]/2)
			SigmatR = Sigmat(xe[i] + h[i]/2)

			SigmaaL = Sigmaa(xe[i] - h[i-1]/2)
			SigmaaR = Sigmaa(xe[i] + h[i]/2)

			ap[i] = mu2[i]/(SigmatL*h[i-1]) + mu2[i]/(SigmatR*h[i]) + \
				1/3*(SigmaaL*h[i-1] + SigmaaR*h[i]) 

		# build right coefficients 
		ae = np.zeros(N) 
		for i in range(N-1):

			# evaluate cross sections at cell centers 
			SigmatR = Sigmat(xe[i] + h[i]/2)
			SigmaaR = Sigmaa(xe[i] + h[i]/2)

			ae[i] = -1/(SigmatR*h[i])*mu2[i+1] + 1/6*SigmaaR*h[i] 

		# boundary conditions 
		# reflecting left BC 
		ap[0] = mu2[0]/(Sigmat(xe[0] + h[0]/2)*h[0]) + 1/3*Sigmaa(xe[0] + h[0]/2)*h[0]
		# marshak right BC
		ap[-1] = mu2[-1]/(Sigmat(xe[-1] - h[-1]/2)*h[-1]) + \
			1/3*Sigmaa(xe[-1] - h[-1]/2)*h[-1] + .5

		# make public 
		self.N = N 
		self.h = h 
		self.aw = aw 
		self.ap = ap 
		self.ae = ae 
		self.xe = xe 

	def solve(self, q):

		assert (len(q) == self.N)

		b = np.zeros(self.N)

		for i in range(1, self.N-1):

			b[i] = self.h[i-1]*(1/6*q[i-1] + 1/3*q[i]) + self.h[i]*(1/6*q[i+1] + 1/3*q[i])

		b[0] = self.h[0]*(1/3*q[0] + 1/6*q[1])
		b[-1] = self.h[-1]*(1/3*q[-1] + 1/6*q[-2]) 

		phi = TDMA(self.aw, self.ap, self.ae, b)

		return self.xe, phi 

if __name__ == '__main__':
	from scipy.interpolate import interp1d 

	Sigmaa = .1 
	Sigmat = .83 

	xb = 50

	Q = 1

	N = 100 # number of cells 
	xe = np.linspace(0, xb, N+1)
	mu2 = np.ones(N+1)/3 
	fem = FEM(xe, mu2, lambda x: Sigmaa, lambda x: Sigmat)
	x, phi = fem.solve(np.ones(N+1)*Q)

	# exact solution 
	D = 1/(3*Sigmat) 
	L = np.sqrt(D/Sigmaa)
	c1 = -Q/Sigmaa/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	phi_ex = lambda x: c1*np.cosh(x/L) + Q/Sigmaa 
	x_ex = np.linspace(0, xb, 100)

	# plt.plot(x_ex, phi_ex(x_ex), '--')
	# plt.plot(x, phi)
	plt.plot(x, np.fabs(phi - phi_ex(x)))
	plt.yscale('log')
	plt.show()

	# check order of convergence 
	N = np.array([25, 50, 100, 200])
	fe = [FEM(np.linspace(0, xb, x), np.ones(x)/3, 
		lambda x: Sigmaa, lambda x: Sigmat) for x in N]

	err = np.zeros(len(N))
	for i in range(len(N)):

		x, phi = fe[i].solve(np.ones(N[i])*Q)

		err[i] = np.linalg.norm(phi - phi_ex(x), 2)

		# f = interp1d(x, phi)
		# xeval = 9*xb/10
		# err[i] = np.fabs(f(xeval) - phi_ex(xeval))

		plt.plot(x, np.fabs(phi - phi_ex(x)))

	plt.yscale('log')
	plt.show()

	fit = np.polyfit(np.log(xb/N), np.log(err), 1)

	print(fit[0])

	print(np.log(err[0]/err[1])/np.log(2))

	plt.loglog(xb/N, np.exp(fit[1]) * (xb/N)**fit[0], '-o')
	plt.loglog(xb/N, err, '-o')
	plt.show()