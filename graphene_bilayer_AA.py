import numpy as np
from ase import *
from ase.visualize import view
from gpaw import GPAW, PW, FermiDirac
from ase.optimize import QuasiNewton, BFGS
from ase.constraints import UnitCellFilter

vdw = 'vdW-DF'
a = 2.46
b = a / np.sqrt(3)
d = 3.35
grap_bilayer = Atoms('C'*36,
                 positions=[[0,     0, 0],
                            [0,     b, 0],
                            [0,     0, d],
                            [0,     b, d],
                            [a,     0, 0],
                            [a,     0, d],
                            [a,     b, 0],
                            [a,     b, d],
                            [2*a,   0, 0],
                            [2*a,   0, d],
                            [2*a,   b, 0],
                            [2*a,   b, d],
                            [0.5*a, 1.5*b, 0],
                            [0.5*a, 1.5*b, d],
                            [0.5*a, 2.5*b, 0],
                            [0.5*a, 2.5*b, d],
                            [1.5*a, 1.5*b, 0],
                            [1.5*a, 1.5*b, d],
                            [1.5*a, 2.5*b, 0],
                            [1.5*a, 2.5*b, d],
                            [2.5*a, 1.5*b, 0],
                            [2.5*a, 1.5*b, d],
                            [2.5*a, 2.5*b, 0],
                            [2.5*a, 2.5*b, d],
                            [a, 3*b, 0],
                            [a, 3*b, d],
                            [a, 4*b, 0],
                            [a, 4*b, d],
                            [2*a, 3*b, 0],
                            [2*a, 3*b, d],
                            [2*a, 4*b, 0],
                            [2*a, 4*b, d],
                            [3*a, 3*b, 0],
                            [3*a, 3*b, d],
                            [3*a, 4*b, 0],
                            [3*a, 4*b, d]],
                            
                 cell=[[      3*a,       0,     0],
                       [ 1.5* a, 4.5 * b, 0],
                       [      0,       0, 10 * a]],
                 
                 pbc=True)

grap_bilayer.append(Atom('Li', (0.5*a , 0.5 * b, 0.5 * d)))
grap_bilayer.append(Atom('Li', (2*a , 2 * b, 0.5 * d)))
grap_bilayer.append(Atom('Li', (3.5*a , 3.5 * b, 0.5 * d)))

name = "graphene"
calc = GPAW(mode=PW(700),
                 xc=vdw,
                 txt= name+'.txt',
                 convergence={'bands':-15},
                 kpts={'density': 4, 'gamma': True},
                 occupations=FermiDirac(width=0.05))
grap_bilayer.set_calculator(calc)

sf = UnitCellFilter(grap_bilayer,mask=[1,1,1,0,0,0]) # here you relax also the lattice. In your script, you relax only the atom positions
opt = BFGS(sf,trajectory =name+'.traj', logfile = name+'.log', restart = name+'.pckl')
opt.run(fmax=0.05)
calc.write(name+'.gpw')
view(grap_bilayer)