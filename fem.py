#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from TDMA import * 

class FEM:

	def __init__(self, N, Sigmaa, Sigmat, xb, BCL=0, BCR=0):

		D = 1/(3*Sigmat) 

		dx = xb/N 

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

		phi = TDMA(-self.aW, self.ap, -self.aE, q)

		return self.x, phi 

if __name__ == '__main__':

	fem = FEM(100, .1, .83, 1)

	x, phi = fem.solve(np.ones(100))

	plt.plot(x, phi)
	plt.show()

