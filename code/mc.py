#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import os 

import ProgressBar as pb # progress bar 

import Timer 

def montecarlo(NPS, Nb, N, Sigmat, Sigmas, Sigmaa, Q, xb=3, seed=1907348632):
	''' Monte Carlo lineatrons
		reflecting on left, vacuum on right 
		Inputs:
			NPS: number of particles sampled per batch
			Nb:  number of batches 
			N:  number of spatial cells 
			Sigmat:  total cross section (1/cm)
			Sigmas:  scattering cross section (1/cm)
			Sigmaa:  absorption cross section (1/cm)
			Q:  source strength (part/cm-s)
			xb:  domain length (cm)
			seed: random seed 
	''' 

	tt = Timer.timer() # start timer 

	NPS = int(NPS) # convert to integer 

	# set random seed for repeatability 
	np.random.seed(seed) 

	# --- set up spacial discretization --- 
	edges = np.linspace(0, xb, N+1) # cell  boundaries 
	dx = xb/N # grid spacing 
	centers = np.linspace(dx/2, xb - dx/2, N) # cell centers 


	# --- set up tallies --- 
	leakR = np.zeros(Nb) # number that leak out right boundary 
	leakL = np.zeros(Nb) # left leakage 
	absorb = np.zeros((Nb, N)) # number of absorptions per cell 
	flux = np.zeros((Nb, N)) # store track lengths in each cell 
	flux2 = np.zeros((Nb,N)) # store square of tracklenghts in each cell 


	# --- transport --- 
	bar = pb.progressbar(max=Nb, time=True) # initialize progress bar 
	for i in range(Nb): # loop through batches 
		for j in range(NPS): # loop through particles per patch 
			x = xb*np.random.rand() # initial starting position

			cell = 0 # stores cell location of particle 

			# find cell of starting particle 
			for k in range(len(edges)):
				if (x > k*dx and x <= (k+1)*dx):

					cell = k 

			# sample direction 
			right = 1 
			if (np.random.rand() >= .5): # set to left 
				right = 0 

			alive = 1 # alive if hasn't leaked or absorbed 
			while (alive):

				startx = x # store position before transport 

				s = -1/Sigmat * np.log(np.random.rand()) # distance to collision

				# update position, right = 1 or 0 
				x += right*s - (1-right)*s 

				# if past the right boundary of the cell 
				if (x > edges[cell+1] and right==1):

					# if leaked out of rightmost cell 
					if (cell == N-1):
						assert(x >= edges[-2]) # assert in last cell 

						alive = 0 # kill particle 
						leakR[i] += 1 # update leak tally 
						flux[i,cell] += xb - startx 

					# if leak from inner cell 
					else:
						flux[i,cell] += edges[cell+1] - startx # distance traveled in cell 

						# update position and cell 
						x = edges[cell+1] # set x to cell boundary of cell 
						cell += 1 # move to cell one to the right 

				# if past left boundary of cell 
				elif (x < edges[cell] and right==0):

					# if leaked from first cell 
					if (cell == 0):
						flux[i,cell] += startx - 0 # distance traveled in cell 

						right = 1 # switch directions 
						leakL[i] += 1 # update leak tally 

						# update position and cell 
						x = 0 # reflect starting at x = 0 
						cell = 0 # explicitly set cell to be correct 

					# if leak from inner cell 
					else:
						flux[i,cell] += startx - edges[cell] 
						x = edges[cell] # set x to left cell boundary 
						cell -= 1 # move to cell on left 

				# if still in cell 
				else: 
					# assert particle inside cell 
					assert(x >= edges[cell])
					assert(x <= edges[cell+1]) 

					flux[i,cell] += s # track length inside cell 

					if (np.random.rand() < Sigmas/Sigmat): # scatter 

						# set direction 
						if (np.random.rand() <= .5): # set to rightward 
							right = 1 
						else: # set to left 
							right = 0 

					else: # absorbed 
						absorb[i,cell] += 1 # tally absorption 
						alive = 0 # kill particle 

		bar.update() # update progress bar after each batch  

		# normalize tallies 
		absorb[i,:] *= Q*xb/(NPS*dx) # absorptions/cm-s
		flux[i,:] *= Q*xb/(NPS*dx) # part-cm/cm-s

		leakL[i] *= Q*xb/NPS # part/s 
		leakR[i] *= Q*xb/NPS # part/s 


	# --- process tallies --- 
	# lambda function to calculate deviation 
	sigma = lambda x: np.sqrt(np.mean(x**2) - np.mean(x)**2)/np.sqrt(Nb-1)

	leakL_bar = np.mean(leakL) # average of left leakage 
	leakR_bar = np.mean(leakR) # average of right leakage 

	s_leakL = sigma(leakL) # deviation in left leakage
	s_leakR = sigma(leakR) # deviation in right leakage 

	flux_bar = np.zeros((N,2)) # store average fluxes and std in each cell 
	absorb_bar = np.zeros((N,2)) # stores average and std absorption rate
	for i in range(N):
		flux_bar[i,0] = np.mean(flux[:,i]) # average the flux in each cell from all batches 
		# deviation of flux 
		flux_bar[i,1] = sigma(flux[:,i])

		# average absorption rate
		absorb_bar[i,0] = np.mean(absorb[:,i])
		# deviation of absorption rate
		absorb_bar[i,1] = sigma(absorb[:,i])

	tt.stop() # end timer

	return centers, flux_bar, np.array([leakL_bar, s_leakL]), np.array([leakR_bar, s_leakR])

if __name__ == '__main__':
	import texTools as tex 
	from hidespines import * 
	# materials
	Sigmat = 1 # cm^-1
	Sigmas = .5 # cm^-1 
	Sigmaa = Sigmat - Sigmas
	Q = 1 # p/cm-s 
	xb = 3 # cm 

	NPS = 1000 # particles per batch 
	Nb = 10 # batches 
	N = 10 # spatial cells 

	# run MC 
	centers, flux, leakL, leakR = montecarlo(NPS, Nb, N, Sigmat, Sigmas, Sigmaa, Q, xb)

	print('Left Leak =', leakL[0], leakL[1])
	print('Right Leak =', leakR[0], leakR[1])

	x = np.linspace(0, xb, 100)
	alpha = np.sqrt(Sigmaa*Sigmat)
	a = -Sigmat*Q/(Sigmaa*(Sigmat*np.cosh(alpha*xb) + alpha*np.sinh(alpha*xb)))
	phi_ex = a*np.cosh(alpha*x) + Q/Sigmaa 

	rl = -1/Sigmat * alpha*a * np.sinh(alpha*xb)

	print('Right Leak Exact =', rl/Q)

	print('MC within', np.fabs(leakR[0] - rl)/rl*100, '%')

	# tabulate results 
	table = tex.table()
	for i in range(N):
		table.addLine(
			tex.utils.writeNumber(centers[i]), # center of cell 
			tex.utils.writeNumber(flux[i,0]), # average flux
			tex.utils.writeNumber(flux[i,1], '{:.3e}') # deviation of flux 
			)
	table.save('tex/fluxTable.tex') 

	table = tex.table()
	table.addLine(
		'Left Leakage',
		tex.utils.writeNumber(leakL[0]), # average left leakge 
		tex.utils.writeNumber(leakL[1]), # left leak deviation 
		)
	table.addLine(
		'Right Leakage',
		tex.utils.writeNumber(leakR[0]), # average right leakage 
		tex.utils.writeNumber(leakR[1]), # right leak deviation 
		)
	table.save('tex/leakTable.tex')

	# plot results 
	plt.figure(figsize=(8,6))
	plt.errorbar(centers, flux[:,0], yerr=flux[:,1], label='MC')
	plt.plot(x, phi_ex, label='Exact')
	plt.xlabel('x')
	plt.ylabel(r'$\phi$')
	plt.legend(loc='best', frameon=False)
	hidespines(plt.gca())
	plt.savefig('tex/flux.pdf')
	plt.show()
