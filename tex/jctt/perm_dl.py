#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 

sys.path.append('../../code')

import ld as LD 

from exactDiff import exactDiff

import texTools as tex 

''' compare the permuations of linear representation in diffusion limit ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
	ftype = '.' + outfile.split('.')[1]
	outfile = sys.argv[1].split('.')[0]
else:
	outfile = None 

Nruns = 5
eps = np.logspace(-5, -1, Nruns)
print(eps)

tol = 1e-6

def getIt(eps, opt, gauss):

	N = 50

	xb = 10

	x0 = np.linspace(0, xb, N+1)
	Sigmat = lambda x: 1 
	Sigmaa = lambda x: .1 
	q = lambda x: 1

	n = 8 

	it = np.zeros(len(eps))

	diff = np.zeros(len(eps))

	phi = np.zeros((len(eps), N))

	plt.figure()

	for i in range(len(eps)):

		sol = LD.Eddington(x0, n, lambda x: eps[i], lambda x: 1/eps[i], 
			lambda x, mu: eps[i], OPT=opt, GAUSS=gauss)

		x, phi[i,:], it[i] = sol.sourceIteration(tol, maxIter=200)

		phi_ex = exactDiff(eps[i], 1/eps[i], eps[i], xb)

		diff[i] = np.linalg.norm(phi[i,:] - phi_ex(x), 2)/np.linalg.norm(phi_ex(x), 2)

	# plt.plot(x, phi[0,:], label='$\epsilon=$' + '{:.2e}'.format(eps[0]))
	plt.plot(x, phi[0,:], label='VEF')
	phi_ex = exactDiff(eps[0], 1/eps[0], eps[0], xb)
	plt.plot(x, phi_ex(x), '--', label='Analytic Diffusion')
	plt.xlabel(r'$x$ (cm)')
	plt.ylabel(r'$\phi(x)$ (1/cm$^2$-s)')
	plt.legend()
	plt.savefig('figs/profile.pdf')

	return diff, it 

diff1, it0 = getIt(eps, 2, 1)
diff0, it1 = getIt(eps, 3, 1)

plt.figure()
plt.loglog(eps, it0, '--', clip_on=False, label='Flat')
plt.loglog(eps, it1, '-', clip_on=False, label='van Leer')

plt.xlabel(r'$\epsilon$')
plt.ylabel('Number of Iterations')
plt.legend()
# if (outfile != None):
# 	plt.savefig(outfile+ftype)

table = tex.table() 
for i in range(Nruns-1, -1, -1):
	table.addLine(
		tex.utils.writeNumber(eps[i], '{:.1e}'), 
		str(int(it0[i])), 
		str(int(it1[i]))
		)
if (outfile != None):
	table.save(outfile+ftype)

plt.figure()
plt.loglog(eps, diff0, '--', clip_on=False, label='Flat')
plt.loglog(eps, diff1, '-', clip_on=False, label='van Leer')

plt.xlabel(r'$\epsilon$')
plt.ylabel('Error')
plt.legend()
# if (outfile != None):
# 	plt.savefig(outfile+'1'+ftype)
# else:
# 	plt.show()

plt.show()