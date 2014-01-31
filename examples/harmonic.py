"""Demonstrates gpe1d module for finding the ground state of 
a simple harmonic potential."""

import numpy as np
from gpe1d.gpe1d import gpe1d_python
import matplotlib.pyplot as plt

from scipy.constants import pi, hbar

def Prob(psi):
    return psi*np.conj(psi)

# Using typical parameters in cold atom experiments
mass = 1.4e-25      # Mass of 87Rb
omega = 2.0*pi*150  # 150 Hz trapping frequency

length_scale = (hbar/2.0/mass/omega)**0.5
time_scale = 1.0/omega
energy_scale = hbar*omega

# we shall work in units of the harmonic oscillator strength
X = np.linspace(-10,10, 10000)
dx = X[1] - X[0]
U = 0.5*X**2

psi_guess = np.exp(-X**2)
psi_analytical = (pi)**-0.25*np.exp(-X**2/2.0)
psi_disp = (pi)**-0.25*np.exp(-(X-2.0)**2/2.0)
# normalize it

(K, T, psi_ground, ep) = gpe1d_python(epsilon=1.0, kappa=0.0, N=2000, 
                            k=0.001, X=X, U=U, psi0=psi_guess, Ntstore=10, 
                            imag_time=1, error=0.0002)
# look at evolution of a displaced ground state
(K, T, psi_out, ep) = gpe1d_python(epsilon=1.0, kappa=0.0, N=2000, 
                            k=0.001, X=X, U=U, psi0=psi_disp, Ntstore=10, 
                            imag_time=0, error=0.0002)



# Plot the ground state of the wavefunction
plt.figure(1)
plt.subplot(211)
plt.title('Finding ground state of a harmonic oscillator')
plt.plot(X, Prob(psi_guess))
plt.plot(X, Prob(psi_ground))
plt.plot(X, Prob(psi_analytical))
plt.legend(['guess', 'imag. time', 'analytical'])

plt.subplot(212)
plt.title('Simulation of a displaced ground state')
plt.plot(X, Prob(psi_out.T))
plt.show()

