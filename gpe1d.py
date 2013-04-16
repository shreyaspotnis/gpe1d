"""A module which provides numerical routines to solve the one 
dimensional Gross Pitaevski equation.

Exported functions:
gpe1d - Numerically solve the 1d GPE using time splitting fourier spectral 
        method.

Copyright 2012 Shreyas Potnis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.fft as fft

def gpe1d(epsilon, kappa, N, k, X, U,  psi0, Ntstore=10, imag_time = 0, 
            error = 0.0002):
    """gpe1d - Numerically solve the 1d GPE using time splitting fourier
    spectral method.
    
    gpe1d(epsilon, kappa, N, k, X, U,  psi0, Ntstore=10, imag_time = 0, 
            error = 0.0002):

    Numerically solves dimensionless GPE equation:
        i*epsilon dpsi/dt = (-epsilon*2/2 d^2/dx^2 + U(x) + kappa|psi|^2)psi
        see W. Boa et al. / Journal of Computational Physics 187 (2003) 318-342
        for details. 

    Arguments:
    epsilon - (a0/xs)^2 where a0 is the ground state harmonic oscillator length
    and xs is the size of the condensate. For no interactions, epsilon = 1, 
    whereas for higher interactions, makes sense to choose a higher epsilon.

    kappa - characterizes the strength of the interactions, for the 1D case,
    this depends on the transverse trapping frequencies
    
    N - number of time steps to take

    k - size of a time step
    
    X - numpy 1d array, grid of all X values

    U - numpy 1d array, 1D potential, usually taken to be 0.5*X**2

    psi0 - complex numpy 1d array, the initial condition for the simulation

    Ntstore - number of psi profiles to store as the simulation progresses.
    These are stored at equally spaced time intervals
    
    imag_time - whether or not to propagate in imaginary time. Propagate in
    imaginary time to find the ground state of the given potential. 

    error - if propagating in imaginary time, this sets the absolute error
    threshold. If the error is psi goes below this, the solution is assumed 
    to have  converged and the simulation stops.

    Returns:
    K - numpy 1d array, grid of k values computed using fftfreq. Useful while 
    analyzing fft of psi.
    
    T - numpy 1d array, the time steps for which psi is stored
    
    psi_out - complex numpy 2d array, size Ntstore*size(X). The output of the
    simulators at different times.

    ep - useless if real time propagation. For imaginary time propagation, gives
    the absolute error in the wavefunction. Useful to see how the solution
    converges.
    
    """
    Ntskip = N/(Ntstore-1)
    h = X[1]-X[0]   # h is the X mesh spacing
    M = np.size(X)  # Number of points in the X grid
    K = fft.fftfreq(M,h)*2.0*np.pi  # k values, used in the fourier spectrum 
                                    # analysis
    T = np.zeros(Ntstore)
    if imag_time == 0:
        prefactor = 1j
        psi_out = np.zeros((Ntstore,M),complex)
        psi_out[0,:] = psi0
        ep = 0
    else:
        prefactor = 1
        ep = np.zeros(Ntstore)
        psi_out = np.zeros(M,complex)
        psiprev = psi0

    U1 = -prefactor*U/2.0/epsilon*k
    C1 = -prefactor*kappa*k/2.0/epsilon
    Kin = np.exp(-prefactor*epsilon*K**2*k/2.0)
    psi = psi0

    for t1 in range(Ntstore-1):
        for t2 in range(Ntskip):
            # Split the entire time stepping into three steps.
            # The first is stepping by time k/2 but only applying the potential
            # and the mean field parts of the unitary
            psi =  np.exp(U1+C1*psi*np.conj(psi))*psi
            # The second part is applying the Kinetic part of the unitary. This
            # is done by taking the fourier transform of psi, so applying this
            # unitary in k space is simply multiplying it by another array
            psi = fft.ifft(Kin*fft.fft(psi))
            # The third part is again stepping by k/2 and applying the potential
            # part of the unitary
            psi =  np.exp(U1+C1*psi*np.conj(psi))*psi
            if imag_time:
                # If we are propagating in imaginary time, then the solution
                # dies down, we need to explicitly normalize it
                psi_int = np.sum(np.conj(psi)*psi)*h
                psi = psi/psi_int**0.5

        # Store the wavefuction in psi_out
        T[t1+1] = (t1+1)*k*Ntskip
        if imag_time==0:
            psi_out[t1+1,:] = psi
        else:    
            # Calculate how much the solution has converged.
            ep[t1+1] = np.sum(np.abs(psiprev-psi))/M
            psiprev = psi
            if ep[t1+1]<=error:
                # If the absolute error is less than specified, stop now.
                psi_out = psi
                break
    return (K,T,psi_out)

