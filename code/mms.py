#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD
import dd as DD 

from scipy.interpolate import interp1d 

''' Test MMS functions in LD and DD ''' 

def getOrder(sol, N):

	tol = 1e-6

	print('Method =', sol[0].name)

	phi_mms = lambda x: np.sin(np.pi*x/xb) # exact solution 

	err = np.zeros(len(sol))
	for i in range(len(sol)):

		sol[i].setMMS()
		# make video 
		# x, phi, it = sol[i].sourceIteration(tol, PLOT='phi' + str(N[i]))
		x, phi, it = sol[i].sourceIteration(tol)

		phi_int = interp1d(x, phi)

		err[i] = np.fabs(phi_mms(xb/2) - phi_int(xb/2))/phi_mms(xb/2)

	# 	plt.plot(x, phi, '-o')

	# plt.show()

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0], fit[1])

	return err

N = np.array([40, 80, 100, 160])

n = 8 

Sigmaa = lambda x: .1
Sigmat = lambda x: 1

xb = 2

dd = [DD.DD(np.linspace(0, xb, x), n, Sigmaa, 
	Sigmat, np.ones((n,x-1)), BCL=0, BCR=1) for x in N]

dd_edd = [DD.Eddington(np.linspace(0, xb, x), n, Sigmaa, 
	Sigmat, np.ones((n,x-1)), BCL=0, BCR=1) for x in N]

ld = [LD.LD(np.linspace(0, xb, x), n, Sigmaa, 
	Sigmat, np.ones((n,x-1)), BCL=0, BCR=1) for x in N] 

ld_edd = [LD.Eddington(np.linspace(0, xb, x), n, Sigmaa, 
	Sigmat, np.ones((n,x-1)), BCL=0, BCR=1, CENT=0) for x in N] 

# eddington using centers 
ld_edd2 = [LD.Eddington(np.linspace(0, xb, x), n, Sigmaa, 
	Sigmat, np.ones((n,x-1)), BCL=0, BCR=1, CENT=1) for x in N]

# errDD = getOrder(dd, N)
# errDD_edd = getOrder(dd_edd, N)

errLD = getOrder(ld, N)
errLD_edd = getOrder(ld_edd, N)
# errLD_edd2 = getOrder(ld_edd2, N)

# plt.loglog(1/N, errDD, '-o', label='DD')
# plt.loglog(1/N, errDD_edd, '-o', label='DD Edd')
plt.loglog(1/N, errLD, '-o', label='LD')
plt.loglog(1/N, errLD_edd, '-o', label='LD Edd')
# plt.loglog(1/N, errLD_edd2, '-o', label='LD Edd2')
plt.legend(loc='best')
plt.show()