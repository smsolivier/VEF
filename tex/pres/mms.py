#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d 

from hidespines import * 

import sys 

sys.path.append('../../code')

import ld as LD 

''' Test MMS functions in LD and DD ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

def getOrder(sol, N):

	tol = 1e-12

	print('Method =', sol[0].name)

	phi_mms = lambda x: np.sin(np.pi*x/xb) # exact solution 

	err = np.zeros(len(sol))
	for i in range(len(sol)):

		sol[i].setMMS()

		x, phi, it = sol[i].sourceIteration(tol, 1000)

		phi_int = interp1d(x, phi)

		# err[i] = np.fabs(phi_mms(xb/2) - phi_int(xb/2))/phi_mms(xb/2)
		err[i] = np.linalg.norm(phi_mms(x) - phi, 2)/np.linalg.norm(phi_mms(x), 2)

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	print(fit[0], fit[1])

	return err

N = np.array([40, 80, 160, 320, 640])

n = 8 

Sigmaa = lambda x: .1
Sigmat = lambda x: 1

xb = 1

# make solver objects 
ld = [LD.LD(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n, x)), BCL=0, BCR=1) for x in N]

ld_edd = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, np.ones((n,x)), BCL=0, BCR=1) for x in N]

# get order of accuracy 
errLD = getOrder(ld, N)
errLD_edd = getOrder(ld_edd, N)

plt.loglog(xb/N, errLD, '-o', label='Unaccelerated')
plt.loglog(xb/N, errLD_edd, '-o', label='Accelerated')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$h$', fontsize=20)
plt.ylabel('Error', fontsize=20)
hidespines(plt.gca())
if (outfile != None):
	plt.savefig(outfile, transparent=True)
else:
	plt.show()