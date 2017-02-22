#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ClosedMethods as cm 

import ld 

n = 4

sigmat = 1 
c = .3
sigmaa = sigmat*(1 - c)

xb = 1
x = np.linspace(0, xb, 100)

q = 10

mu, w = np.polynomial.legendre.leggauss(n)

def nufunc(nu):

	x = 0
	for i in range(n):

		x += nu*w[i]/(nu - mu[i])

	return x - 1/c


eps = 1e-3
nu = np.zeros(n)

nu[0] = cm.bisection(nufunc, -10, -1)

nu[-1] = -1*nu[0]

for i in range(1, int(n/2)):

	nu[i] = cm.bisection(nufunc, mu[i-1]+eps, mu[i]-eps)

	nu[n - i - 1] = -nu[i]

print(nu)

print(nufunc(nu))

psi = lambda x, mu, nu: c*nu/(nu - mu)*np.exp(sigmat*(xb/2 - x)/nu)

A = np.zeros((int(n/2),int(n/2)))
for i in range(int(n/2)):

	for j in range(int(n/2)):

		A[i,j] = psi(x[-1], mu[i], nu[j])

b = np.zeros(int(n/2)) - q/sigmaa

alpha_neg = np.linalg.solve(A, b)

for i in range(int(n/2), n):

	ii = i - int(n/2)

	for j in range(int(n/2),n):

		jj = j - int(n/2)

		A[ii,jj] = psi(x[0], mu[i], nu[j])

alpha_pos = np.linalg.solve(A, b)

alpha = np.concatenate((alpha_neg, alpha_pos))

PSI = np.zeros((n, len(x)))
for i in range(n):

	for j in range(n):

		PSI[i,:] += psi(x, mu[i], nu[j])*alpha[j]

phi = np.zeros(len(x))

for i in range(n):

	phi += w[i] * PSI[i,:]

sn = ld.LD(np.linspace(0, xb, 50), n, lambda x: sigmaa, lambda x: sigmat, np.ones((n,50))*q, BCL=1, BCR=1)

xsn, phisn, it = sn.sourceIteration(1e-6)

plt.plot(x, phi - phi[0] + phisn[0])
plt.plot(xsn, phisn)
plt.show()