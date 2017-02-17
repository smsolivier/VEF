#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N = 25

xb = 1 
x = np.linspace(0, xb, N)

dx = xb/N

psiR = np.zeros((2,N))
psiL = np.zeros((2,N)) 

sigmat = .1

# sweep left to right (mu > 0)
psi0R = 0 # vacuum 
A = np.zeros((2,2))

for i in range(1, N):

	# left 
	A[0,0] = 1/2 + sigmat*dx/2 
	A[0,1] = 1/2 

	# right 
	A[1,0] = -1/2 
	A[1,1] = -1/2 + sigmat*dx/2 + 1

	# rhs 
	b = np.ones(2)
	b[0] += psiR[0,i-1]

	ans = np.linalg.solve(A, b)

	psiL[0,i] = ans[0] 
	psiR[0,i] = ans[1] 

psiL[1,-1] = psiR[0,-1] 
for i in range(N-2, -1, -1):

	# left 
	A[0,0] = 1/2 + sigmat*dx/2 + 1
	A[0,1] = -1/2

	# right
	A[1,0] = 1/2
	A[1,1] = 1/2 + sigmat*dx/2 

	b = np.ones(2)
	b[1] += psiL[1,i+1]

	ans = np.linalg.solve(A, b)

	psiL[1,i] = ans[0] 
	psiR[1,i] = ans[1]

psi = 1/2*(psiL[0,:] + psiR[0,:]) + 1/2*(psiL[1,:] + psiR[1,:])

plt.plot(x, psi)
plt.show()