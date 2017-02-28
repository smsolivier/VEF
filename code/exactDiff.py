#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

def refMarshak(Sigmaa, Sigmat, Q, xb):

	D = 1/(3*Sigmat) 
	L = np.sqrt(D/Sigmaa)
	c1 = -Q/Sigmaa/(np.cosh(xb/L) + 2*D/L*np.sinh(xb/L))
	phi_ex = lambda x: c1*np.cosh(x/L) + Q/Sigmaa 

	return phi_ex 

def bothMarshak(Sigmaa, Sigmat, Q, xb):
	''' Analytic solution for diffusion with double 
		marshak conditions at -xb and xb 
	''' 

	D = 1/(3*Sigmat)
	L = np.sqrt(D/Sigmaa) 

	A = np.zeros((2, 2))

	A[0,0] = np.sinh(-xb/L) - 2*D/L*np.cosh(-xb/L)
	A[0,1] = np.cosh(-xb/L) - 2*D/L*np.sinh(-xb/L) 
	A[1,0] = np.sinh(xb/L) + 2*D/L*np.cosh(xb/L)
	A[1,1] = np.cosh(xb/L) + 2*D/L*np.sinh(xb/L) 

	b = np.ones(2)*-Q/Sigmaa 

	c = np.linalg.solve(A, b) 

	phi = lambda x: c[0] * np.sinh(x/L) + c[1] * np.cosh(x/L) + Q/Sigmaa

	return phi 

def exactDiff(Sigmaa, Sigmat, Q, xb, BCL=0, BCR=1):

	if (BCL == 0 and BCR == 1):

		phi = refMarshak(Sigmaa, Sigmat, Q, xb) 

	elif (BCL == 1 and BCR == 1):

		phi = bothMarshak(Sigmaa, Sigmat, Q, xb)

	else:

		print('\n--- WARNING: BC not defined in exactDiff.py ---\n')
		sys.exit()

	return phi 

if __name__ == '__main__':

	phi = exactDiff(.1, .83, 1, 1)

	x = np.linspace(-1, 1, 20)

	plt.plot(x, phi(x))
	plt.show()



