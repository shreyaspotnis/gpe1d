import numpy as np
import os
import deltakick as dk

scratch = os.environ['SCRATCH']
fzr = np.array([20., 40., 60., 80., 100., 120., 140])
fzr = np.array([20.])

for fz in fzr:
    filename = scratch + '/fz' + str(fz)+'.run2.dat' 
    dk.loadParms('parms.ini')
    dk.fz = fz
    dk.init()
    dk.printExpParms()
    dk.simulate()
    dk.saveData(filename)
