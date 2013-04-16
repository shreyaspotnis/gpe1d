"""A class to handle delta-kick cooling calculations. This is so that I can
easily automate running over multiple parameters."""

import sys
import os
import argparse
import ConfigParser
import shelve

import numpy as np
import numpy.fft as fft

import gpe1d_c as gp

# Fundamental constants
pi = 3.14159
hbar = 1.054571628e-34  # (SI) Reduced Planck's constant
h = hbar*2*pi           # (SI) Planck's constant
kb = 1.3806e-23         # (SI) Boltzman constant
c = 2.99792458e08       # (m/s) speed of light
mu0 = 4*pi*1e-07        # (SI) Permeability of Vaccuum
ep0 = 8.854187e-12      # (SI) Permittivity of Vaccuum

# Rubidium constants
m_rb = 1.443e-25         # (kg) Mass of Rubiduim87
lambda_780 = 780.281e-09 # (m) Vaccuum wavelength of Rb87 2S1/2 to 2P3/2
lambda_1064 = 1064e-09   # (m) Wavelength of the ODT
a_bg = 5.1e-09           # (m) Rubidium87 background scattering length
alpha_0 = 0.0794         # (h.Hz/(V/cm)^2) Ground state polarizability
gamma = 2.0*pi*6e6       # (Hz) Linewidth of the Rubidium transition
k_1064 = 2.0*pi/lambda_1064 
e_recoil_1064 = hbar**2*k_1064**2/2.0/m_rb 
t_recoil_1064 = e_recoil_1064/kb
v_recoil_1064 = hbar*k_1064/m_rb
k_780 = 2*pi/lambda_780
k_1064 = 2*pi/lambda_1064
w_780 = c*k_780
w_1064 = c*k_1064
p_recoil = hbar*k_780
e_recoil = p_recoil**2/(2.0*m_rb)
t_recoil = e_recoil/kb

# Let there be gravity
g = 9.81                # (m/s^2) we are on earth

# list all the experimental parameters
expt_parms = {
             'fx':float,
             'fy':float,
             'fz':float,
             'n_atoms': float,
             'expand_time': float,
             'cooling_time':float
             }

sim_parms = {'size': int,
             'dt': float,
             'nt_imag': int,
             'dx': float,
             'nt_store_expand': int,
             'nt_store_cool': int,
             'length_scale':float
             }

def loadParms(parmsfile):
    parms = ConfigParser.SafeConfigParser()
    parms.read(parmsfile)
    for ep, eptype  in expt_parms.items():
        globals()[ep] = eptype(parms.get('expt',ep))
    for sp, sptype in sim_parms.items():
        globals()[sp] = sptype(parms.get('sim',sp))

def loadVars(varnames):
    for vname, vval in varnames:
        if vname in sim_parms:
            globals()[vname] = sim_parms[vname](vval)
        elif vname in expt_parms:
            globals()[vname] = expt_parms[vname](vval)
        else:
            print vname, 'not found, ignoring'

def loadData(filename):
    datashelf = shelve.open(filename)
    for key in datashelf:
        #if key == 'psi_cool':
        #    continue
        globals()[key] = datashelf[key]
        print key

def saveData(filename):
    out_parms = ('K2', 'T1', 'Tfree', 'Tcool', 'psi_ground', 'psi_free',
                    'psi_cool')
    # this means that we have to save everything
    datashelf = shelve.open(filename, 'n')

    for dicts in (sim_parms, expt_parms, out_parms):
        for key in dicts:
            try:
                datashelf[key] = globals()[key]
            except TypeError:
                print('ERROR shelving: {0}'.format(key))
    datashelf.close()

def Prob(psi):
    """Return the probability distribution of psi."""
    return np.real(psi*np.conj(psi))

def TrapLattice(X, trap_center, lattice_depth, lattice_period):
    U_temp =  0.5*(X-trap_center)**2 + lattice_depth*np.sin(pi*X/lattice_period)**2
    u_center = 0.5*trap_center**2
    return U_temp - u_center

def TiltedLattice(X, lattice_depth, lattice_period, force):
    return lattice_depth*np.sin(pi*X/lattice_period)**2 - force*X 

def simulate():
    global K2, T1, Tfree, Tcool, psi_ground, psi_free, psi_cool
    # Propagate through imaginary time to find the ground state
    (K1, T1, psi_ground) = gp.gpe1d(epsilon, kappa, nt_imag, 
                                        dt, X, U_0, psi_0, nt_store_expand, 1)
    # Free Expansion
    (K2, Tfree, psi_free) = gp.gpe1d(epsilon, kappa, nt_expand, dt, X, 
                            U_0*0.0, psi_0, nt_store_expand, 0)
   
    psi_cool = []
    for psif in psi_free:
        (K2, Tcool, psi_2) = gp.gpe1d(epsilon, kappa, nt_cooling, dt, X, 
                                  U_0, psif, nt_store_cool, 0)
        psi_cool.append(psi_2)
def plotstuff():
    import matplotlib.pyplot as plt
    # get the filenames first
    file_ground = args.load + '.ground.png'
    file_delv = args.load + '.delv.png'

    plt.figure(1)
    plt.clf()
    plt.plot(x_real*1e6, np.real(psi_0*np.conj(psi_0)))
    plt.plot(x_real*1e6, psi_ground*np.conj(psi_ground))
    plt.legend(('guess', 'imag time'))
    plt.xlim((-z_3d*4*length_scale*1e6, z_3d*4*length_scale*1e6))
    plt.xlabel('x/(um)')
    plt.savefig(file_ground)

    indices = np.arange(len(t_free_real))

    plt.figure(2)
    plt.clf()
    plt.subplot(311)
    plt.plot(t_free_real*1e3, delv_ex_cool[:,0]*1e3)
    plt.ylim((0, delv_ex_cool[:,0].max()*1e3*1.2))
    for ind in indices:
        t_start = t_free_real[ind]
        t_cooling = t_start + t_cool_real
        plt.plot(t_cooling*1e3, delv_ex_cool[ind, :]*1e3)
    plt.ylabel('$\Delta v (mm/s)$')
    plt.xlabel('$t(ms)$')

    plt.subplot(312)
    plt.plot(t_free_real*1e3, delx_ex_cool[:,0]*1e6)
    plt.ylim((0, delx_ex_cool[:,0].max()*1e6*1.2))
    for ind in indices:
        t_start = t_free_real[ind]
        t_cooling = t_start + t_cool_real
        plt.plot(t_cooling*1e3, delx_ex_cool[ind, :]*1e6)
    plt.ylabel('$\Delta x (\mu m)$')
    plt.xlabel('$t(ms)$')

    plt.subplot(313)
    plt.plot(t_free_real*1e3, meanfield_ex_cool[:,0]/kb*1e9)
    plt.ylim((0, meanfield_ex_cool[:,0].max()/kb*1e9*1.2))
    for ind in indices:
        t_start = t_free_real[ind]
        t_cooling = t_start + t_cool_real
        plt.plot(t_cooling*1e3, meanfield_ex_cool[ind, :]/kb*1e9)
    plt.ylabel('$MFE (nK)$')
    plt.xlabel('$t(ms)$')
    plt.savefig(file_delv)


    plt.show()


def analyze():
    global dk, x_real, t_free_real, t_cool_real
    global k_real, v_real, e_real, temp_real
    global psi_k_free, psi_k_cool
    global psi2_k_free, psi2_k_cool
    global psi2_free, psi2_cool
    global x_ex_cool, x2_ex_cool, p_ex_cool, p2_ex_cool
    global delx_ex_cool, delp_ex_cool, kin_ex_cool
    global v_ex_cool, v2_ex_cool, delv_ex_cool
    global psi_ground, psi_cool, psi_free
    global meanfield_ex_cool, psi2_orig

    
    # Calculate stuff in real units
    dk = K2[1]-K2[0]
    x_real = X*length_scale    # (m)
    t_free_real = Tfree*time_scale # (s)
    t_cool_real = Tcool*time_scale # (s)
    t_free_real = np.linspace(0, expand_time/fz, nt_store_expand)
    t_cool_real = np.linspace(0, cooling_time/fz, nt_store_cool)
    k_real = K2/length_scale
    v_real = hbar*k_real/m_rb
    e_real = hbar**2*k_real**2/2.0/m_rb
    temp_real = e_real/kb

    # convert psi_cool to a numpy array
    psi_cool = np.array(psi_cool)

    psi2_free = np.real(psi_free*np.conj(psi_free))
    psi2_cool = np.real(psi_cool*np.conj(psi_cool))

    # get momentum space wavefunction
    psi_k_free = fft.fft(psi_free)
    psi_k_cool = fft.fft(psi_cool)

    # normalize it
    psi_k_free *= (dx/dk/nx)**0.5
    psi_k_cool *= (dx/dk/nx)**0.5

    psi2_k_free = np.real(psi_k_free*np.conj(psi_k_free))
    psi2_k_cool = np.real(psi_k_cool*np.conj(psi_k_cool))

    # Normalize to sum, not integral
    sum2 = np.sum(psi2_free, -1)
    psi2_free /= sum2[..., None]
    sum2 = np.sum(psi2_cool, -1)
    sum_cool = sum2
    psi2_orig = psi2_cool.copy()
    psi2_cool /= sum2[..., None]
    sum2 = np.sum(psi2_k_free, -1)
    psi2_k_free /= sum2[..., None]
    sum2 = np.sum(psi2_k_cool, -1)
    psi2_k_cool /= sum2[..., None]

    x_ex_cool = np.sum(psi2_cool*x_real, -1)
    x2_ex_cool = np.sum(psi2_cool*x_real*x_real, -1)
    p_ex_cool = np.sum(psi2_k_cool*hbar*k_real,-1)
    p2_ex_cool = np.sum(psi2_k_cool*hbar*hbar*k_real*k_real,-1)
    meanfield_ex_cool = np.sum(psi2_cool**2, -1)*sum_cool*kappa*energy_scale/2.0
    meanfield_ex_cool = np.sum(psi2_orig**2, -1)/sum_cool*kappa*energy_scale/2.0

    delx_ex_cool = (x2_ex_cool-x_ex_cool**2)**0.5
    delp_ex_cool = (p2_ex_cool-p_ex_cool**2)**0.5
    kin_ex_cool = p2_ex_cool/2.0/m_rb
    v_ex_cool = p_ex_cool/m_rb
    v2_ex_cool = p2_ex_cool/m_rb**2
    delv_ex_cool = delp_ex_cool/m_rb

def init():
    """Initialize global variables before the simulation."""
    global nx, lx, wx, wy, wz, wpx
    global wpy, wpz, time_scale, energy_scale
    global u_0, delta, epsilon, gamma_x, gamma_y, kw1, kw2, kw3, ks1, ks2, ks3
    global weak2d, weak1d, mu_3d, z_3d, x_3d, y_3d, mu_1d, z_1d, X
    global nt_expand, nt_cooling

    # calculated parameters
    nx = 2**size    # Number of grid points
    lx = dx*nx      # Calculate the length of the x-domain
    # Calculate non-interacting gaussian widths
    wx = (hbar/(m_rb*2.0*pi*fx))**0.5
    wy = (hbar/(m_rb*2.0*pi*fy))**0.5
    wz = (hbar/(m_rb*2.0*pi*fz))**0.5

    wpx = (hbar*m_rb*2*pi*fx)**0.5 # (SI) Momentum width
    wpy = (hbar*m_rb*2*pi*fy)**0.5
    wpz = (hbar*m_rb*2*pi*fz)**0.5

    # Choose the length scale of the simulation
    time_scale = m_rb*wz**2/hbar
    energy_scale = m_rb*length_scale**2/time_scale**2 # (kg m^2/s^2)
    
    # Setup simulation parameters
    u_0 = 4.0*pi*hbar**2*a_bg/m_rb
    delta = 4.0*pi*a_bg*n_atoms/wz
    epsilon = (wz/length_scale)**2
    gamma_x = fx/fz
    gamma_y = fy/fz

    kw1 = delta*epsilon**1.5*(gamma_x*gamma_y)**0.5/2.0/pi
    kw2 = delta*epsilon**2*(gamma_y/2.0/pi)**0.5
    kw3 = delta*epsilon**2.5

    ks1 =((delta*gamma_y*gamma_x)**(3.0/5.0)*
          epsilon**1.5*pi/9.0*(15.0/4.0/pi)**(8.0/5.0))
    ks2 =((delta*gamma_y)**(4.0/5.0)*epsilon**2.0*
          (4.0*pi/15.0/gamma_x)**(1.0/5.0)*(5.0/7.0))
    ks3 = delta*epsilon**(5.0/2.0)

    weak2d = delta*gamma_y**0.5
    weak1d = delta*(gamma_x*gamma_y)**0.5

    # First find the approximate 3d Thomas-Fermi wavefuction
    mu_3d = epsilon/2.0*(15.0*delta*gamma_y*gamma_x/4.0/pi)**(2.0/5.0)
    z_3d = (2.0*mu_3d)**0.5
    x_3d = (2.0*mu_3d)**0.5/gamma_x
    y_3d = (2.0*mu_3d)**0.5/gamma_y
    # The 3d function basically follows the shape of the potential in the ellipsoid
    # regoin in which mu-V>0. Integrating over the other two axes, we get the
    # density distribution in the axis that we are concerned with

    # The approximate ground state wavefuction for the 1d gpe
    mu_1d = (3.0/2.0*ks1)**(2.0/3.0)/2.0
    z_1d = (2.0*mu_1d)**0.5
    
    X = (np.arange(0,nx)-(nx-1)/2.0)*dx

    nt_expand = int(expand_time/fz/time_scale/dt)
    nt_cooling = int(cooling_time/fz/time_scale/dt)

    initU()     # Initialize potentials
    initPsi()

def initU():
    global U_0, U_1
    """Initialize potentials."""
    # Initialize arrays and psi
    U_0 = 0.5*X**2

def initPsi():
    """Initialize wavefunctions."""
    global psi_0, psi_3d_tf, psi_1d_tf, amp_ni, psi_ni, kappa

    psi_0 = np.zeros(nx,complex)
    psi_3d_tf = np.zeros(nx,complex)
    psi_1d_tf = np.zeros(nx,complex)
    for i in range(nx):
       if(mu_3d-U_0[i]>0.0):
            psi_3d_tf[i] = (pi/gamma_x/gamma_y/ks3)**0.5*(mu_3d-U_0[i])
       if(mu_1d-U_0[i]>0.0):
            psi_1d_tf[i] = ((mu_1d-U_0[i])/ks1)**0.5

    amp_ni = 1.0/(pi*epsilon)**0.25 # normalization
    psi_ni = amp_ni*np.exp(-(X/length_scale)**2/epsilon/2.0)

    if args.nointeraction:
        psi_0 = psi_ni
        kappa = 0
    else:
        psi_0 = psi_1d_tf
        kappa = ks1

def printExpParms():
    # Calulate some barrier related stuff
    s_trap_f = "Trap Frequencies fx:{0:.2f}Hz fy:{1:.2f}Hz fz:{2:.2f}Hz".\
            format(fx,fy,fz)
    s_NI = "HO: wx={0:.3f}um wy={1:.3f}um wz={2:.3f}um".\
            format(wx*1e06,wy*1e06,wz*1e06)
    s_TF = "Thomas-Fermi widths: wx={0:.3f}um wy={1:.3f}um wz={2:.3f}um".\
            format(x_3d*length_scale*1e06,y_3d*length_scale*1e06,
                   z_3d*length_scale*1e06)
    s_length = "Total Length:\t{0:.2f} mm".format(dx*nx*length_scale*1e3)

    # Print experimental parameters
    print "-----Experimental parameters-----"
    print s_trap_f
    print s_NI
    print s_TF
    # Print some simulation parameters
    print "-----Simulation Parameters-----"
    print "Length Scale:\t{0:.2f} um".format(length_scale*1e6)
    print "Total Length:\t{0:.2f} mm".format(dx*nx*length_scale*1e3)
    print "Length Step:\t{0:.2f} um".format(dx*length_scale*1e6)
    print "Time Scale:\t{0:.2f} ms".format(time_scale*1e3)
    print "Expand Time Length:\t{0:.2f} ms".format(dt*nt_expand*time_scale*1e3)
    print "Cooling Time Length:\t{0:.2f} ms".format(dt*nt_cooling*time_scale*1e3)
    print "Time Step:\t{0:.2f} us".format( dt*time_scale*1e6)
    print "\nSpatial Steps:\t", nx
    print "Expand Time steps:\t", nt_expand
    print "Cooling Time steps:\t", nt_cooling
    print "r (dt/dx**2):\t", dt/dx**2

if __name__ == "__main__":
     # Setup the argument parser, as we will be running this as a script
    parser = argparse.ArgumentParser(
              description='Simulate the dynamics of a BEC in a one dimensional waveguide.')
    parser.add_argument('--var', nargs= 2, action='append', 
                            metavar=('VARIABLENAME', 'VALUE'),
                            help='''Change an expermintal parameter. Useful while
                            running the program multiple times. For example, to
                            change the number of atoms, use --var n_atoms 2000
                            ''')
    parser.add_argument('-s', '--save', metavar=('PREFIX'), 
                            help='''Save all data, and prefix all filenames with
                            the given string PREFIX''')
    parser.add_argument('-l', '--load', metavar='filename',
                            help='''Load the results of a simulation and plot
                            graphs''')
    parser.add_argument('--parms', type=str, metavar='parmfile.ini', 
                            default='parms.ini',                         
                            help='''Use this for all the default simulation
                            parameters. Useful when running the script repeatedly
                            while changing only a few parameters. Put all your
                            parameters in PARMS, and pass only the ones you
                            want to change using -f.
                            ''')
    parser.add_argument('--nosim', dest='simulate', action='store_const',
                             const=False, default=True,
                            help='''Do not run the simulator. Can be used with the 
                            run -i option in ipython''')
    parser.add_argument('--noplot', dest='plot', action='store_const',
                            const=False, default=True,
                            help='Do not plot the results of the simulation.')
    parser.add_argument('--info', action='store_const', const=True,
                            default=False,
                            help='''Just print information on the simulation parameters
                            and exit. Useful while playing with simulation parameters
                            and not wanting to run the whole simulation''')
    parser.add_argument('--nointeraction', action='store_const', const=True,
                            default=False,
                            help='''Switch off interactions.''')
    args = parser.parse_args()
    
    # see whether a parms file was supplied
    if args.parms:
        loadParms(args.parms)

    # check to see if the --var option has been used
    if args.var:
        loadVars(args.var)
    
    # check if we have to load data from a file
    if args.load:
        loadData(args.load)
        # we dont want to run the simulator if we are loading from a file
        # just plot everything
        args.simulate = False 

    # initialize global variables
    init()
    printExpParms()
    # check to see if we were asked to just print info and quit
    if args.info:
        # yes, we were, sa quit
        sys.exit(0)
    if args.simulate:
        simulate()
    if args.load:
        # We analyze the data if it was loaded
        analyze()
        plotstuff()
    if args.save:
        saveData(args.save)
else:
    # Module was not loaded as a script, do some initialization here
    class Namespace:
        def __init__(self):
            self.nointeraction = False

    args = Namespace()
