### hlimit.py
compares difference between Sn and moment equations as cell width --> 0 
### direct.py
Direct S2 solver 
### mhfem_acc.py
Mixed Hybrid Finite Element solver for moment equations (drift diffusion) 
### converge.py
plot eddington and flux convergence for unaccelerated and accelerated Sn 
### transport.py
General Discrete Ordinates class 

Provides common variable names and helpful functions 

Parent class for all spatial discretization methods and acceleration methods
### mms.py
Test MMS functions in LD and DD 
### checkAccel.py
compare the number of iterations for unaccelerated, Eddington, and S2SA S8 
### ld.py
Linear Discontinuous Galerkin spatial discretization of Sn 

Inherits from transport.py 

Has unaccelerated, eddington accelerated methods
### exactDiff.py
computes analytic diffusion solutions 
### diffMMS.py
find order of accurracy of LD and DD Eddington acceleration in the diffusion limit 
### Timer.py
Class for timing functions 
### hidespines.py
Make plots matplotlib plots look nice 
### dvs.py
compares diffusion and transport 
### fem.py
Regular finite element diffusion solver 
### dd.py
Diamond differenced Sn solver 

Inherits from transport.py 

Includes unaccelerated, eddington accelerated, S2SA, DSA
### diffLimit.py
compare number of iterations in LD and DD Eddington Acceleration 

in the Diffusion Limit (epsilon --> 0)
