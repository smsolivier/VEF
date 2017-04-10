# Eddington Acceleration Python Code
Code for implementing Eddington Acceleration into Linear Discontinuous Galerkin Sn 


## ld.py
Lumped Linear Discontinuous Galerkin spatial discretization of Sn 

Inherits from transport.py 

Includes 

* Unaccelerated

* Eddington Accelerated 


## mhfem_acc.py
Mixed Hybrid Finite Element solver for moment equations (drift diffusion) 

## dd.py
Diamond differenced Sn solver 

Inherits from transport.py 

Includes 

* unaccelerated 

* eddington accelerated 

* S2SA 

* inconsistent DSA


## direct.py
Direct S2 solver 

## transport.py
General Discrete Ordinates class 

Provides common variable names and helpful functions 

Parent class for all spatial discretization methods and acceleration methods 


## fem.py
Regular finite element diffusion solver 

## mms.py
Test order of accuracy for LD options 

## hlimit.py
compares difference between Sn and moment equations as cell width --> 0 

## perm_dl.py
compare the permuations of linear representation in diffusion limit 

## diffLimit.py
compare number of iterations in LD and DD Eddington Acceleration 

in the Diffusion Limit (epsilon --> 0) 


## checkAccel.py
compare the number of iterations for unaccelerated, Eddington, and S2SA S8 

## exactDiff.py
computes analytic diffusion solutions 

## diffMMS.py
find order of accurracy of LD and DD Eddington acceleration in the diffusion limit 

## testLD.py
Plot error from MMS solution for LD 

## reed.py
Test LD with Reed's Problem 

## Timer.py
Class for timing functions 

## hidespines.py
Make matplotlib plots look nice 

