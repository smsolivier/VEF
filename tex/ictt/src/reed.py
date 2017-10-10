#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../../code')

import ld as LD

if (len(sys.argv) > 1):
	outname = sys.argv[1]
else:
	outname = None

Sigmamax = 10
Sigmat = lambda x: Sigmamax*(x<2) + .001*(x>=2)*(x<4) + \
	1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + 1*(x>=7)*(x<=8)
Sigmaa = lambda x: Sigmamax*(x<2) + .1*(x>=4)*(x<6) + 5*(x>=6)*(x<7) + .1*(x>=7)*(x<=8) 
Q = lambda x, mu: Sigmamax*(x<2) + 1*(x>=7)*(x<=8)

N = 1000
n = 8
xb = 8 
tol = 1e-6
x = np.linspace(0, xb, N+1)

s2 = LD.S2SA(x, n, Sigmaa, Sigmat, Q)
ed = LD.Eddington(x, n, Sigmaa, Sigmat, Q, OPT=3, GAUSS=1)

x, phi, it = ed.sourceIteration(tol)

plt.figure(figsize=(16, 5.25))
line = 1.75
plt.plot(x, phi)
plt.xlim(0, xb)
plt.ylim(0, line)
plt.annotate('High Source, Absorption', 
	xy=(1, line),
	horizontalalignment='center',
	verticalalignment='top',
	fontsize=18
	)
plt.axvline(2, color='k', alpha=.3)
plt.annotate('Pseudo Void', 
	xy=(3, line),
	horizontalalignment='center',
	verticalalignment='top',
	fontsize=18
	)
plt.axvline(4, color='k', alpha=.3)
plt.annotate(r'$c=.9$', 
	xy=(5, line),
	horizontalalignment='center',
	verticalalignment='top',
	fontsize=18
	)
plt.axvline(6, color='k', alpha=.3)
plt.annotate(r'$c=1$', 
	xy=(6.5, line),
	horizontalalignment='center',
	verticalalignment='top',
	fontsize=18
	)
plt.axvline(7, color='k', alpha=.3)
plt.annotate(r'Source, $c=.9$', 
	xy=(7.5, line),
	horizontalalignment='center',
	verticalalignment='top',
	fontsize=18
	)
plt.xlabel(r'$x$ (cm)')
plt.ylabel(r'$\phi$ (1/cm$^2$-s)')

if (outname != None):
	plt.savefig(outname)
else:
	plt.show()