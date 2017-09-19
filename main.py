from matplotlib.pyplot import plot, xlabel, ylabel, show

import HFSolver
import numpy as np
from pylab import *

# Fixed values that do not change while running the program
dt = np.float
nucleiPositions = [(-1), (-1), (-1), (-1), (1), (1), (1), (1)]
# create a 1x8 Matrix, for easier use with the indices
base = [(13.00773), (1.962079), (0.444529), (0.1219492)]*2
numberOfElectrons = 2
breakValue = 1e-8
maximumIterations = 400 # Hard bound for the SCL, to not lock my computer up if it diverges!

hfSolver = HFSolver.HFSolver(base, nucleiPositions, numberOfElectrons, breakValue, maximumIterations)

# equilibrium position in a.u.
R_AB = (1.3983)
# Guess a initial rho (simple => set to zero)
rho = np.zeros((len(base), len(base)), dtype=dt)


(energy, rho) = hfSolver.SCF(rho, R_AB)

print energy

rRange = arange(0.5, 4., 0.1)  # Pylab function, float range!
Er = []
for r in rRange:
    Er.append(hfSolver.SCF(rho, r)[0])
plot(rRange, Er, 'r--')
xlabel('R_AB')
ylabel('Energy')
show()