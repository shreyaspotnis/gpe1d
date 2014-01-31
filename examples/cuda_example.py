import numpy as np
#import matplotlib.pyplot as plt
from time import time 
from gpe1d.gpe1d import gpe1d_python, gpe1d_cuda
import sys

def Prob(psi):
    return psi*np.conj(psi)

def Prob(psi):
    return psi*np.conj(psi)

size = 16
nx = 2**size
dx = 1.0/32.0 
dt = 0.0005
Nt = 10000
X = (np.arange(0,nx)-(nx-1)/2.0)*dx
U = 0.5*X**2
psi0 = np.exp(-X**2/10)+1j*0
psi0 = psi0/(np.sum(psi0*np.conj(psi0))*dx)**0.5
epsilon = 0.16 
kappa = 52 

print "Nx = ", nx, ", Nt = ", Nt

print "Running cuda"
start_time = time()
(K,T,psi_out_c) = gpe1d_cuda(epsilon, kappa, Nt, dt, X, U,  psi0, Ntstore=10,
                                imag_time = 1)
end_time = time()
print "Time taken: ", end_time-start_time

print "Running python"

start_time = time()
(K,T,psi_out,ep) = gpe1d_python(epsilon, kappa, Nt, dt, X, U,  psi0, Ntstore=10,
                                imag_time = 1)
end_time = time()
print "Time taken: ", end_time-start_time

psi2 = np.conj(psi_out)*psi_out
psi2_c = np.conj(psi_out_c)*psi_out_c

psi_std_c = np.std(Prob(psi2-psi2_c))
psi_mean = np.mean(Prob(psi2))
psi_max_c = np.max(Prob(psi2-psi2_c))

print "STD/MEAN: cuda", psi_std_c/psi_mean
print "MAX: cuda", psi_max_c
