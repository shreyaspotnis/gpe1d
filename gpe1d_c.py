"""A module which provides various numerical routines to solve the one dimensional Gross Pitaevski equation.

Exported functions:
gpe1d - Numerically solve the 1d GPE using time splitting fourier spectral method.

"""

import numpy as np
import numpy.fft as fft
import sys
import os

def gpe1d(epsilon, kappa, N, k, X, U,  psi0, Ntstore=10, imag_time=0,
            kernel='./gpec_cuda',
            infile='/dev/shm/gpec_input', outfile='/dev/shm/gpec_output'):
    def write_int(stream, i):
        stream.write('%d\n' % i)
    def write_float(stream, f):
        stream.write('%.17g\n' % f)
    def write_complex_vec(stream, vec):
        for v in vec:
            stream.write('%.17g ' % v.real)
            stream.write('%.17g ' % v.imag)
        stream.write('\n')
    def write_real_vec(stream, vec):
        for v in vec:
            stream.write('%.17g ' % v)
        stream.write('\n')

    Ntskip = N/(Ntstore-1)
    h = X[1]-X[0]   # h is the X mesh spacing
    M = np.size(X)  # Number of points in the X grid
    K = fft.fftfreq(M,h)*2.0*np.pi  # k values, used in the fourier spectrum analysis
    T = np.zeros(Ntstore)

    if imag_time:
        prefactor = 1.0
        psi_out = np.zeros(M,complex)
    else:
        prefactor = 1.0j
        psi_out = np.zeros((Ntstore,M),complex)
        psi_out[0,:] = psi0

    U1 = np.exp(-prefactor*U/2.0/epsilon*k)
    c2 = -kappa*k/2.0/epsilon
    Kin = np.exp(-prefactor*epsilon*K**2*k/2.0)/M

    # IMPORTANT: we are dividing Kin by M, as the fftw and ifftw routines
    # are not normalized - ie ifft(fft(A)) would give is M*A, where M is
    # the number of elements in A. Hence, we need to divide by M

    f = open(infile, 'w')

    # send all this data to our c simulator
    # write Nx, N, epsilon, kappa, k, X, U, psi0, Ntstore
    write_int(f, M)
    write_int(f, Ntstore)
    write_int(f, Ntskip)
    write_complex_vec(f, psi0)
    write_complex_vec(f, U1)
    write_complex_vec(f, Kin)
    write_float(f, c2)
    write_float(f, h)
    write_int(f, imag_time)

    f.close()

    os.system(kernel + ' < ' + infile + ' > ' + outfile)

    f = open(outfile, 'r')

    if imag_time:
        psi_out = eval(f.readline())
    else:
        for t1 in range(Ntstore-1):
            psi_out[t1+1,:] = eval(f.readline())
            T[t1+1] = (t1+1)*k*Ntskip

    f.close()

    return (K,T,psi_out)

