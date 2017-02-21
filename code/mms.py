#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from ld import *
from dd import *

from scipy.interpolate import interp1d 

''' Test MMS functions in LD and DD ''' 

xb = 2
Sigmaa = lambda x: .1 
Sigmat = lambda x: .83 

n = 4

tol = 1e-6

N = np.array([20, 40, 80])
mms = [DD(np.linspace(0, xb, x), n, Sigmaa, Sigmat, np.ones((n,x)), BCL=1, BCR=1) for x in N]

phi_mms = lambda x: np.sin(np.pi*x/xb)

err = np.zeros(len(mms))
for i in range(len(mms)):

	mms[i].setMMS()
	x, phi, it = mms[i].sourceIteration(tol)

	phi_int = interp1d(x, phi)

	err[i] = np.fabs(phi_mms(xb/2) - phi_int(xb/2))

fit = np.polyfit(np.log(1/N), np.log(err), 1)

print(fit[0])

plt.loglog(1/N, err, '-o')
plt.show()