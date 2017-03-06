#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mhfem_acc import * 
from exactDiff import * 

import os 
import shutil 

eps = np.logspace(-9, 2, 30)

Sigmaa = .1*eps 
Sigmat = .83/eps 

xb = 1 

Q = 1 * eps 

BCL = 0
BCR = 1 

N = 100
xe = np.linspace(0, xb, N) 
mu2 = np.ones(N)/3 
B = np.ones(N)/2 

# check if video directory exists 
if (os.path.isdir('video')):

	shutil.rmtree('video')

# make video directory 
os.makedirs('video')

# check if output exists 
if (os.path.isfile('output.mp4')):

	os.remove('output.mp4')

for i in range(len(eps)):

	mhfem = MHFEM(xe, lambda x: Sigmaa[i], lambda x: Sigmat[i], BCL, BCR)
	mhfem.discretize(mu2, B)

	x, phi = mhfem.solve(np.ones(N)*Q[i], np.zeros(N), CENT=2)

	phiEdge = mhfem.getEdges(phi)
	phiCent = mhfem.getCenters(phi) 

	xEdge = mhfem.getEdges(x)
	xCent = mhfem.getCenters(x) 

	phi_ex = exactDiff(Sigmaa[i], Sigmat[i], Q[i], xb, BCL, BCR)

	# plt.figure()
	# plt.subplot(1,2,1)
	# plt.plot(x, phi, '-o', label='MHFEM')
	# plt.plot(x, phi_ex(x), label='Exact')
	# plt.xlabel('x')
	# plt.ylabel(r'$\phi$')
	# plt.legend(loc=1)
	# plt.ylim(0, 1.5)

	# plt.subplot(1,2,2)
	plt.figure()
	plt.semilogy(xEdge, np.fabs(phiEdge - phi_ex(xEdge))/phi_ex(xEdge), '-o', label='Edge')
	plt.semilogy(xCent, np.fabs(phiCent - phi_ex(xCent))/phi_ex(xCent), '-o', label='Center')
	# plt.semilogy(x, np.fabs(phi - phi_ex(x)), '-o')
	plt.xlabel('x')
	plt.ylabel('| MHFEM - Exact | / Exact ')
	plt.legend(loc=2)
	# plt.ylim(1e-6, 1e-2)
	plt.title(r'$\epsilon = $' + '{:.5e}'.format(eps[i]))

	# plt.show()

	plt.savefig('video/' + str(i) + '.png')
	plt.close()

os.system('ffmpeg -f image2 -r 1 -i video/%d.png -b 320000k output.mp4')
