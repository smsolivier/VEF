#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD
import dd as DD 

from scipy.interpolate import interp1d 

from hidespines import * 

from R2 import * 

import texTools as tex

''' Test order of accuracy for LD options ''' 

if (len(sys.argv) > 1):
	outfile = sys.argv[1] 
else:
	outfile = None 

# use MMS solution with mu in it 
OPT = 1 

def getOrder(sol, N, tol, label):

	print('Method =', sol[0].name)

	phi_mms = lambda x: np.sin(np.pi*x/xb) # exact solution 

	err = np.zeros(len(sol))
	for i in range(len(sol)):

		sol[i].setMMS(OPT)
		# make video 
		# x, phi, it = sol[i].sourceIteration(tol, PLOT='phi' + str(N[i]))
		x, phi, it = sol[i].sourceIteration(tol, 200)

		phi_int = interp1d(x, phi)

		# err[i] = np.fabs(phi_mms(xb/2) - phi_int(xb/2))/phi_mms(xb/2)
		err[i] = np.linalg.norm(phi_mms(x) - phi, 2)/np.linalg.norm(phi_mms(x), 2)

	# 	plt.plot(x, phi, '-o')

	# plt.show()

	fit = np.polyfit(np.log(1/N), np.log(err), 1)

	# fit equation 
	f = lambda x: np.exp(fit[1]) * x**(fit[0])

	# R^2 value
	r2 = rsquared(err, f(1/N))

	print(fit[0], np.exp(fit[1]), r2)

	plt.loglog(xb/N, err, '-o', clip_on=False, label=label)

	return err, fit[0], np.exp(fit[1]), r2

# N = np.array([80, 160, 320, 640, 1280])
N = np.logspace(1.2, 3, 5)
N = np.array([int(x) for x in N])

n = 8 

Sigmaa = lambda x: .01
Sigmat = lambda x: 1

xb = 1

print(xb/N)

q = lambda x, mu: 1

tol = 1e-6

# make solver objects 
# flat 
ed0 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=3, GAUSS=1) for x in N]

# use edges 
ed1 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa,
	Sigmat, q, OPT=1, GAUSS=1) for x in N]

# van leer 
ed2 = [LD.Eddington(np.linspace(0, xb, x+1), n, Sigmaa, 
	Sigmat, q, OPT=2, GAUSS=1) for x in N]

# get order of accuracy 
size = 2
err = np.zeros((size, len(N)))
order = np.zeros(size)
b = np.zeros(size)
r = np.zeros(size)
reconstruct = ['Flat', 'van Leer']
# gauss = ['Average', 'Rational Polynomial', 'Average', 'Rational Polynomial']

# err[0,:], order[0], b[0], r[0] = getOrder(ed00, N, tol, 'No Slopes, No Gauss')
# err[1,:], order[1], b[1], r[1] = getOrder(ed01, N, tol, 'No Slopes, Gauss')
# err[2,:], order[2], b[2], r[2] = getOrder(ed10, N, tol, 'Slope from Edges, No Gauss')
# err[3,:], order[3], b[3], r[3] = getOrder(ed11, N, tol, 'Slopes from Edges, Gauss')
# err[2,:], order[2], b[2], r[2] = getOrder(ed20, N, tol, 'vanLeer, No Gauss')
# err[3,:], order[3], b[3], r[3] = getOrder(ed21, N, tol, 'vanLeer, Gauss')

err[0,:], order[0], b[0], r[0] = getOrder(ed0, N, tol, 'Flat')
# err[1,:], order[1], b[1], r[1] = getOrder(ed1, N, tol, 'Edge')
err[1,:], order[1], b[1], r[1] = getOrder(ed2, N, tol, 'Van Leer')

# plt.loglog(xb/N, err20[-1]/(xb/N[-1])**2*(xb/N)**2, 
# 	color='k', alpha=.7, label='Slope = 2')
# plt.legend(loc='best', frameon=False)
# plt.xlabel(r'$h$', fontsize=20)
# plt.ylabel('Error', fontsize=20)
# hidespines(plt.gca())
# if (outfile != None):
# 	plt.savefig(outfile, transparent=True)
# else:
# 	plt.show()

# make latex table 
table = tex.table() 
for i in range(len(reconstruct)):

	table.addLine(
		reconstruct[i], 
		tex.utils.writeNumber(order[i], '{:.4}'),
		tex.utils.writeNumber(b[i], '{:.3}'),
		tex.utils.writeNumber(r[i], '{:.4e}')
		)
if (outfile != None):
	table.save(outfile)