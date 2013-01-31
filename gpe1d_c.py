"""A module which provides various numerical routines to solve the one dimensional Gross Pitaevski equation.

Exported functions:
gpe1d - Numerically solve the 1d GPE using time splitting fourier spectral method.

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
import sys
import subprocess 
import struct
from time import time 

def gpe1d(epsilon, kappa, N, k, X, U,  psi0, Ntstore=10, imag_time=0,
            kernel='./gpec_cuda'):
    # start the process
    cuda_process = subprocess.Popen(kernel, 
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE)
    Ntskip = N/(Ntstore-1)
    h = X[1]-X[0]   # h is the X mesh spacing
    M = np.size(X)  # Number of points in the X grid
    K = fft.fftfreq(M,h)*2.0*np.pi  # k values, used in the fourier spectrum analysis
    T = np.zeros(Ntstore)
    if imag_time:
        prefactor = 1.0
    else:
        prefactor = 1.0j
    U1 = np.exp(-prefactor*U/2.0/epsilon*k)
    c2 = -kappa*k/2.0/epsilon
    Kin = np.exp(-prefactor*epsilon*K**2*k/2.0)/M

    # IMPORTANT: we are dividing Kin by M, as the fftw and ifftw routines
    # are not normalized - ie ifft(fft(A)) would give is M*A, where M is
    # the number of elements in A. Hence, we need to divide by M

    # send all the required information
    cuda_process.stdin.write(struct.pack('iiiffi', 
                M, Ntstore, Ntskip, c2, h, imag_time))

    # make 32bit float versions of arrays being sent out 
    psi0_32 = psi0.astype('complex64')
    U1_32 = U1.astype('complex64')
    Kin_32 = Kin.astype('complex64')

    cuda_process.stdin.write(psi0_32.data)
    cuda_process.stdin.write(U1_32.data)
    cuda_process.stdin.write(Kin_32.data)

    start_time = time()
    data_from_cuda = cuda_process.stdout.read()
    print "Read time: ", time()-start_time

    start_time = time()
    if imag_time:
        psi_out_32 = np.ndarray(M, dtype='complex64', buffer=data_from_cuda)
        psi_out = psi_out_32.astype('complex128')
    else:
        psi_out_32 = np.ndarray((Ntstore, M), dtype='complex64', 
                                buffer=data_from_cuda)
        psi_out = psi_out_32.astype('complex128')
    print "Process time :", time() - start_time

    return (K,T,psi_out)
