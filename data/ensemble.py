#!/usr/bin/env python3

#     AbLEF fuses antibody language and structural ensemble representations for property prediction.
#     Copyright © 2023 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""preprocessing script to generate structural ensembles from antibody sequences using ImmuneBuilder and OpenMM"""

import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import torch


# immune builder structure prediciton (similar to igfold, AF2, etc.)
def immune_builder(sequence, output="mAb.pdb"):
    from ImmuneBuilder import ABodyBuilder2
    predictor = ABodyBuilder2()
    antibody = predictor.predict(sequence)
    antibody.save(output)
    

# openmm molecular dynamics
def openmm_implicit(pdb, output= 'mAb.pdb', steps = 50000, T = 300.0, C = 0.1):
    print('')
    print('OPENMM MD w/ IMPLICIT SOLVENT')
    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod= CutoffNonPeriodic,
                                     nonbondedCutoff=2.0*nanometer, constraints=HBonds)
    system = forcefield.createSystem(pdb.topology, soluteDielectric=1.0, solventDielectric=80.0)
    kappa = float(367.434915*sqrt(C/(T*80.0)))
    system = forcefield.createSystem(pdb.topology, implicitSolventKappa=kappa)
    integrator = LangevinMiddleIntegrator(T*kelvin, 1.0/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    # extract set of conformations from simulation
    # create list of steps spaced by 1000 between 0 and steps
    all_steps = np.arange(0, steps, 1000)
    for s in all_steps:
        simulation.reporters.append(PDBReporter('openmm_' + str(s) + '_' + output, s))
        simulation.reporters.append(StateDataReporter(stdout, s, step=True,
                potentialEnergy=True, temperature=True))
        simulation.step(s)
        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open('openmm_' + str(s) + '_' + output, 'w'))  


# if name is main executable
if __name__ == "__main__":
    
    # read arguments
    import argparse
    parser = argparse.ArgumentParser(description='OpenMM MD w/ ImmuneBuilder PDBs')
    parser.add_argument('--h', type=str, required=False, help='heavy chain amino acid sequence')
    parser.add_argument('--l', type=str, required=False,help='light chain amino acid sequence')
    parser.add_argument('--output', type=str, default='mAb.pdb', help='output file name')
    parser.add_argument('--pdb', required=False, type=str, help='path to pdb file')
    parser.add_argument('--T', type=float, default=300.0, help ='temperature')
    parser.add_argument('--conc', type=float, default=0.1, help='concentration')
    parser.add_argument('--steps', type=int, default=50000, help='number of steps')
    args = parser.parse_args()
    
    if args.h and args.l:
        sequence = {
            'H': args.h,
            'L': args.l}
        immune_builder(sequence, args.output)
    elif args.h or args.l:
        print('Error: must provide both heavy and light chain sequences')
        exit()
    elif args.pdb:
        pdb = PDBFile(args.pdb)  
        openmm_implicit(pdb, output=args.output, steps = args.steps, T = args.T, C = args.conc)

