#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from TDMA import * 

''' has trouble with reflective left boundary for large domains ''' 

class FEM:

	def __init__(self, N, Sigmaa, Sigmat, xb, BCL=0, BCR=0):

		D = 1/(3*Sigmat) 

		dx = xb/N 
		self.dx = dx 

		self.x = np.linspace(0, xb, N) 

		# discretize 
		self.aW = np.zeros(N) + D/dx - 1/8*Sigmaa*dx 
		self.aW[0] = 0 

		self.aE = np.zeros(N) + D/dx - 1/8*Sigmaa*dx 
		self.aE[-1] = 0 

		self.ap = np.zeros(N) + 2*D/dx + 3/8*2*Sigmaa*dx 
		self.ap[0] = 3/8*Sigmaa*dx + D/dx 
		self.ap[-1] = D/dx + 1/2 + 3/8*Sigmaa*dx 

	def solve(self, q):

		phi = TDMA(-self.aW, self.ap, -self.aE, q*self.dx)

		return self.x, phi 

if __name__ == '__main__':

	fem = FEM(1000, .1, .83, 100)

	x, phi = fem.solve(np.ones(1000))

	plt.plot(x, phi)
	plt.show()

