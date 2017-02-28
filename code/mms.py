#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ld as LD
import dd as DD 

from scipy.interpolate import interp1d 

''' Test MMS functions in LD and DD ''' 

def getOrder(sol, N):

	xb = 2
	Sigmaa = lambda x: .1 
	Sigmat = lambda x: .83 

	n = 4

	tol = 1e-6

	N = np.array([20, 40, 80])
	mms = [sol(np.linspace(0, xb, x), n, Sigmaa, 
		Sigmat, np.ones((n,x)), BCL=0, BCR=1) for x in N]

	print('Method =', mms[0].name)

	phi_mms = lambda x: np.sin(np.pi*x/xb)

	err = np.zeros(len(mms))
	for i in range(len(mms)):

		mms[i].setMMS()
		x, phi, it = mms[i].sourceIteration(tol)

		phi_int = interp1d(x, phi)

		err[i] = np.fabs(phi_mms(xb/2) - phi_int(xb/2))/phi_mms(xb/2)

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0], fit[1])

	return err

N = np.array([20, 40, 80])

errDD = getOrder(DD.DD, N)
errDD_edd = getOrder(DD.Eddington, N)

errLD = getOrder(LD.LD, N)
errLD_edd = getOrder(LD.Eddington, N)

plt.loglog(1/N, errDD, '-o', label='DD')
plt.loglog(1/N, errDD_edd, '-o', label='DD Edd')
plt.loglog(1/N, errLD, '-o', label='LD')
plt.loglog(1/N, errLD_edd, '-o', label='LD Edd')
plt.legend(loc='best')
plt.show()