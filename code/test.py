#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mc import * 

N = 50

xb = 1 
dx = xb/N
x = np.linspace(dx/2, xb - dx/2, N)

psiR = np.zeros((2,N))
psiL = np.zeros((2,N)) 

sigmat = 1

# sweep left to right (mu > 0)
psi0R = 0 # vacuum 
A = np.zeros((2,2))

# left 
A[0,0] = -1/2 + sigmat*dx/2 + 1
A[0,1] = -1/2

# right
A[1,0] = 1/2
A[1,1] = 1/2 + sigmat*dx/2 

b = np.ones(2)*dx/4

ans = np.linalg.solve(A, b)

psiL[1,-1] = ans[0] 
psiR[1,-1] = ans[1]
for i in range(N-2, -1, -1):

	# left 
	A[0,0] = -1/2 + sigmat*dx/2 + 1
	A[0,1] = -1/2

	# right
	A[1,0] = 1/2
	A[1,1] = 1/2 + sigmat*dx/2 

	b = np.ones(2)*dx/4
	b[1] += psiL[1,i+1]

	ans = np.linalg.solve(A, b)

	psiL[1,i] = ans[0] 
	psiR[1,i] = ans[1]

# bc
A[0,0] = 1/2 + sigmat*dx/2 
A[0,1] = 1/2 
A[1,0] = -1/2 
A[1,1] = -1/2 + sigmat*dx/2 + 1 

b = np.ones(2)*dx/4
b[0] += psiL[1,0] 
ans = np.linalg.solve(A, b)
psiL[0,0] = ans[0]
psiR[0,0] = ans[1] 

for i in range(1, N):

	# left 
	A[0,0] = 1/2 + sigmat*dx/2 
	A[0,1] = 1/2 

	# right 
	A[1,0] = -1/2 
	A[1,1] = -1/2 + sigmat*dx/2 + 1

	# rhs 
	b = np.ones(2)*dx/4
	b[0] += psiR[0,i-1]

	ans = np.linalg.solve(A, b)

	psiL[0,i] = ans[0] 
	psiR[0,i] = ans[1] 

xmc, flux, leakL, leakR = montecarlo(10000, 10, 20, sigmat, 0, .1, 1, xb)

psi = .5*(psiL[0,:] + psiR[0,:]) + .5*(psiL[1,:] + psiR[1,:])
# plt.plot(x, psiL[0,:] + psiR[0,:])
# plt.plot(x, psiL[1,:] + psiR[1,:])
plt.plot(x, psi)
plt.errorbar(xmc, flux[:,0], yerr=flux[:,1])
# plt.plot(psiL[0,:], '-o')
# plt.plot(psiR[0,:], '-o')
plt.show()