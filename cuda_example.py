import numpy as np
#import matplotlib.pyplot as plt
from time import time 
import gpe1d
import gpe1d_c
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
(K,T,psi_out_c) = gpe1d_c.gpe1d(epsilon, kappa, Nt, dt, X, U,  psi0, Ntstore=10,
                                imag_time = 1, kernel='./gpec_cuda')
end_time = time()
print "Time taken: ", end_time-start_time

print "Running python"

start_time = time()
(K,T,psi_out,ep) = gpe1d.gpe1d(epsilon, kappa, Nt, dt, X, U,  psi0, Ntstore=10,
                                imag_time = 1)
end_time = time()
print "Time taken: ", end_time-start_time

sys.exit(0)

psi2 = np.conj(psi_out)*psi_out
psi2_e = np.conj(psi_out_e)*psi_out_e
psi2_v = np.conj(psi_out_v)*psi_out_v
psi2_o = np.conj(psi_out_o)*psi_out_o
psi2_c = np.conj(psi_out_c)*psi_out_c

psi_std_e = np.std(Prob(psi2-psi2_e))
psi_std_v = np.std(Prob(psi2-psi2_v))
psi_std_o = np.std(Prob(psi2-psi2_o))
psi_std_c = np.std(Prob(psi2-psi2_c))
psi_mean = np.mean(Prob(psi2))

psi_max_e = np.max(Prob(psi2-psi2_e))
psi_max_v = np.max(Prob(psi2-psi2_v))
psi_max_o = np.max(Prob(psi2-psi2_o))
psi_max_c = np.max(Prob(psi2-psi2_c))

print "STD/MEAN: vanilla", psi_std_v/psi_mean
print "STD/MEAN: estimate ", psi_std_e/psi_mean
print "STD/MEAN: optimize", psi_std_o/psi_mean
print "STD/MEAN: cuda", psi_std_c/psi_mean

print "MAX: vanilla", psi_max_v
print "MAX: estimate ", psi_max_e
print "MAX: optimize", psi_max_o
print "MAX: cuda", psi_max_c

