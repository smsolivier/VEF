#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from hidespines import * 

import sys 

from scipy.interpolate import interp1d 

sys.path.append('../../code')

import ld as LD 

def make(sol, ot, tol, label):

	err = np.zeros(len(sol))
	it = np.zeros(len(sol))

	for i in range(len(sol)):

		sol[i].setMMS()

		x, phi, it[i] = sol[i].sourceIteration(tol, 1000)

		phi_int = interp1d(x, phi)

		err[i] = np.linalg.norm(sol[i].phi_mms(x) - phi, 2)/ \
			np.linalg.norm(sol[i].phi_mms(x), 2)

	fig = plt.figure()
	twin = fig.twinx()
	fig.loglog(ot, err, '-o', clip_on=False, label=label)
	twin.loglog(ot, it, '-o')

n = 8 

Sigmaa = lambda x: 1 
Sigmat = lambda x: 10

ot = np.logspace(-1, .5, 5)

xb = 1 

N = np.array([int(Sigmat(0)*xb/x) for x in ot])

print(N)

q = lambda x, mu: 1 

tol = 1e-6

ed00 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=0, GAUSS=0) for x in N]

ed01 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=0, GAUSS=1) for x in N]

ed20 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat,q, OPT=2, GAUSS=0) for x in N]

ed21 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=2, GAUSS=1) for x in N]

s2 = [LD.S2SA(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q) for x in N]

make(ed00, ot, tol, 'None, Average')
# make(s2, ot, tol, 'S2SA')
plt.show()