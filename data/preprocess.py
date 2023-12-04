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

# convert pdbs and fastas to padded numpy arrays
from anarci import anarci
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
#from IPython.display import display
import Bio.PDB
from Bio import SeqIO
import time
import glob
import sys
import os
from Bio.PDB import Entity
import json

def center_of_mass(entity, geometric=False):
    """
    Returns gravitic [default] or geometric center of mass of an Entity.
    Geometric assumes all masses are equal (geometric=True)
    """
    
    # Structure, Model, Chain, Residue
    if isinstance(entity, Entity.Entity):
        atom_list = entity.get_atoms()
    # List of Atoms
    elif hasattr(entity, '__iter__') and [x for x in entity if x.level == 'A']:
        atom_list = entity
    else: # Some other weirdo object
        raise ValueError("Center of Mass can only be calculated from the following objects:\n"
                            "Structure, Model, Chain, Residue, list of Atoms.")
    
    masses = []
    positions = [ [], [], [] ] # [ [X1, X2, ..] , [Y1, Y2, ...] , [Z1, Z2, ...] ]
    
    for atom in atom_list:
        masses.append(atom.mass)
        
        for i, coord in enumerate(atom.coord.tolist()):
            positions[i].append(coord)

    # If there is a single atom with undefined mass complain loudly.
    if 'ukn' in set(masses) and not geometric:
        raise ValueError("Some Atoms don't have an element assigned.\n"
                         "Try adding them manually or calculate the geometrical center of mass instead.")
    
    if geometric:
        return [sum(coord_list)/len(masses) for coord_list in positions]
    else:       
        w_pos = [ [], [], [] ]
        for atom_index, atom_mass in enumerate(masses):
            w_pos[0].append(positions[0][atom_index]*atom_mass)
            w_pos[1].append(positions[1][atom_index]*atom_mass)
            w_pos[2].append(positions[2][atom_index]*atom_mass)

        return [sum(coord_list)/sum(masses) for coord_list in w_pos]
    
def calc_residue_dist(residue_one, residue_two) :
    """returns the distance between two residue atoms"""
    diff_vector  = residue_one.coord - residue_two.coord
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

def calc_dist_matrix(atom_one, atom_two, frame) :
    """returns a matrix of atom distances in Fv"""
    answer = numpy.zeros((len(atom_one), len(atom_two)), float)
    for row, residue_one in enumerate(atom_one) :
        for col, residue_two in enumerate(atom_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def calc_oid_dist(residue_one, residue_two) :
    """returns the centroid distance between two residues"""
    diff_vector  = residue_one - residue_two
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

def calc_oid_dist_matrix(oid_one, oid_two, frame) :
    """returns a matrix of centroid distances in Fv"""
    answer = numpy.zeros((len(oid_one), len(oid_two)), float)
    for row, residue_one in enumerate(oid_one) :
        for col, residue_two in enumerate(oid_two) :
            answer[row, col] = calc_oid_dist(residue_one, residue_two)
    return answer

def get_calphas(struct):
    calphas = []
    for residue in struct.get_residues():
        calpha_atoms = []
        resname = str(residue.get_resname())
        if "NME" in resname:
            continue
        for atom in residue.get_atoms():
            atomname = str(atom.get_fullname())
            if "GLY" in resname and "CA" in atomname:
                calpha_atoms.append(atom)
            elif "CA" in atomname:
                calpha_atoms.append(atom)
            else:
                continue
        calphas.append(center_of_mass(calpha_atoms, geometric = False))
    return calphas

def get_cbetas(struct):
    cbetas = []
    for residue in struct.get_residues():
        cbeta_atoms = []
        resname = str(residue.get_resname())
        if "NME" in resname:
            continue
        for atom in residue.get_atoms():
            atomname = str(atom.get_fullname())
            if "GLY" in resname and "CA" in atomname:
                cbeta_atoms.append(atom)
            elif "CB" in atomname:
                cbeta_atoms.append(atom)
            else:
                continue
        cbetas.append(center_of_mass(cbeta_atoms, geometric = False))
    return cbetas

def get_scoids(struct):
    scoids = []
    for residue in struct.get_residues():
        sc_atoms = []
        resname = str(residue.get_resname())
        if "NME" in resname:
            continue
        for atom in residue.get_atoms():
            atomname = str(atom.get_fullname())
            if "GLY" in resname and "CA" in atomname:
                sc_atoms.append(atom)
            elif any([x in atomname for x in not_sc_atoms]):
                continue
            else:        
                sc_atoms.append(atom)
        scoids.append(center_of_mass(sc_atoms, geometric = False))
    return scoids

def get_roids(struct):
    roids = []
    for residue in struct.get_residues():
        roid_atoms = []
        resname = str(residue.get_resname())
        if "NME" in resname:
            continue
        for atom in residue.get_atoms():
            atomname = str(atom.get_fullname())
            if "GLY" in resname and "CA" in atomname:
                roid_atoms.append(atom)
            elif any([x in atomname for x in not_bb_atoms]):
                continue
            else:        
                roid_atoms.append(atom)
        roids.append(center_of_mass(roid_atoms, geometric = False))
    return roids

not_sc_atoms = [" C ", " N ", " O ", " H ", " H1 ", " H2 ", " H3 ", "CA", "HA", "HB", "HD", "HE", "HG", "HZ", "HH"]
not_bb_atoms = [" H ", " H1 ", " H2 ", " H3 ", "HA", "HB", "HD", "HE", "HG", "HZ", "HH"]

def print_figure(matrix, filename):
    """
    Generates a plot of the map's data and saves an image at the given
    filename.
    """
    canvas = FigureCanvas(draw(matrix))
    canvas.print_figure(filename)
    
def init():
    pass

def animate_plot (i):
    pdb_code = pdbs[i].split(".")[0]
    pdb_filename = pdbs[i]
    name = '%s.npy' % (pdb_code)
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    # choose atomset for aLEF
    ## atomset choice
    atoms = numpy.array(get_calphas(structure))
    #atoms = numpy.array(get_cbetas(structure))
    #atoms = numpy.array(get_roids(structure))
    #atoms = numpy.array(get_scoids(structure))
    dist_matrix = calc_oid_dist_matrix(oid_one = atoms, oid_two = atoms, frame = i)
    lngth = len(dist_matrix)

    for i in range(len(lc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, hgth), 0),0)
        lngth, hgth = dist_matrix.shape

    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth), 0),1)
    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth + PAD-lc_len),0),0)

    lngth, hgth = dist_matrix.shape
    hc_anarci_shft = [x + PAD for x in hc_anarci['gaps']]

    for j in range(len(hc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1,lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1, hgth), 0),0)
        lngth = len(dist_matrix)

    lngth = len(dist_matrix)
    dm = numpy.pad(dist_matrix, [(0, pd2mx - lngth), (0, pd2mx - lngth)], mode='constant', constant_values=0)
    
    numpy.save(name, dm)
    contact_map = dist_matrix < 12.0
    cmap = 'plasma'
    cax = ax.imshow(dm
                    , cmap = plt.cm.get_cmap(cmap)
                    , interpolation = 'nearest', origin="lower")
    if i == 0:
        fig.colorbar(cax, cmap = plt.cm.get_cmap(cmap))

    # get cdr indices
    cdr1_PAD, cdr2_PAD, cdr3_PAD = 20, 12, 32
    cdrl1, cdrl2, cdrl3 = lc_anarci['cdr1'], lc_anarci['cdr2'], lc_anarci['cdr3']
    cdrh1, cdrh2, cdrh3 = hc_anarci['cdr1'], hc_anarci['cdr2'], hc_anarci['cdr3']

    #print(cdrl1)
    #print([lc_anarci['seq'][x] for x in cdrl1])
    #print(cdrl2)
    #print([lc_anarci['seq'][x] for x in cdrl2])
    #print(cdrl3)
    #print([lc_anarci['seq'][x] for x in cdrl3])
    #print(cdrh1)
    #print([hc_anarci['seq'][x] for x in cdrh1])
    #print(cdrh2)
    #print([hc_anarci['seq'][x] for x in cdrh2])
    #print(cdrh3)
    #print([hc_anarci['seq'][x] for x in cdrh3])
    
    cdrh1 = [x + PAD for x in cdrh1]
    cdrh2 = [x + PAD for x in cdrh2]
    cdrh3 = [x + PAD for x in cdrh3]

    cdrs = cdrl1 + cdrl2 + cdrl3 + cdrh1 + cdrh2 + cdrh3
    cdr3s = cdrl3 + cdrh3

    # slice cdrs from dist_matrix, insert padding
    cdrs_dist_matrix = dist_matrix[cdrs, :][:, cdrs]
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD +  cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)

    numpy.save('cdrs_'+ name, cdrs_dist_matrix)

    # slice cdr3s from dist_matrix, insert padding
    cdr3s_dist_matrix = dist_matrix[cdr3s, :][:, cdr3s]
    lngth = len(cdr3s_dist_matrix)
    cdr_len = len(cdr3l)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdr3s_dist_matrix)
    cdr3s_dist_matrix = numpy.pad(cdr3s_dist_matrix, [(0, 2*cdr3_PAD - lngth), (0, 2*cdr3_PAD - lngth)], mode='constant', constant_values=0)

    numpy.save('cdr3s_'+ name, cdr3s_dist_matrix)


    return cax

def animate_plot1 (i):
    pdb_code = pdbs[i].split(".")[0]
    pdb_filename = pdbs[i]
    name = '%s.npy' % (pdb_code)
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    # choose atomset for aLEF
    ## atomset choice
    #atoms = numpy.array(get_calphas(structure))
    atoms = numpy.array(get_cbetas(structure))
    #atoms = numpy.array(get_roids(structure))
    #atoms = numpy.array(get_scoids(structure))
    dist_matrix = calc_oid_dist_matrix(oid_one = atoms, oid_two = atoms, frame = i)
    lngth = len(dist_matrix)

    for i in range(len(lc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, hgth), 0),0)
        lngth, hgth = dist_matrix.shape

    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth), 0),1)
    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth + PAD-lc_len),0),0)

    lngth, hgth = dist_matrix.shape
    hc_anarci_shft = [x + PAD for x in hc_anarci['gaps']]

    for j in range(len(hc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1,lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1, hgth), 0),0)
        lngth = len(dist_matrix)

    lngth = len(dist_matrix)
    dm = numpy.pad(dist_matrix, [(0, pd2mx - lngth), (0, pd2mx - lngth)], mode='constant', constant_values=0)
    
    numpy.save(name, dm)
    contact_map = dist_matrix < 12.0
    cmap = 'plasma'
    cax = ax.imshow(dm
                    , cmap = plt.cm.get_cmap(cmap)
                    , interpolation = 'nearest', origin="lower")
    if i == 0:
        fig.colorbar(cax, cmap = plt.cm.get_cmap(cmap))

    # get cdr indices
    cdr1_PAD, cdr2_PAD, cdr3_PAD = 20, 12, 32
    cdrl1, cdrl2, cdrl3 = lc_anarci['cdr1'], lc_anarci['cdr2'], lc_anarci['cdr3']
    cdrh1, cdrh2, cdrh3 = hc_anarci['cdr1'], hc_anarci['cdr2'], hc_anarci['cdr3']

    #print(cdrl1)
    #print([lc_anarci['seq'][x] for x in cdrl1])
    #print(cdrl2)
    #print([lc_anarci['seq'][x] for x in cdrl2])
    #print(cdrl3)
    #print([lc_anarci['seq'][x] for x in cdrl3])
    #print(cdrh1)
    #print([hc_anarci['seq'][x] for x in cdrh1])
    #print(cdrh2)
    #print([hc_anarci['seq'][x] for x in cdrh2])
    #print(cdrh3)
    #print([hc_anarci['seq'][x] for x in cdrh3])
    
    cdrh1 = [x + PAD for x in cdrh1]
    cdrh2 = [x + PAD for x in cdrh2]
    cdrh3 = [x + PAD for x in cdrh3]

    cdrs = cdrl1 + cdrl2 + cdrl3 + cdrh1 + cdrh2 + cdrh3
    cdr3s = cdrl3 + cdrh3

    # slice cdrs from dist_matrix, insert padding
    cdrs_dist_matrix = dist_matrix[cdrs, :][:, cdrs]
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD +  cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)

    numpy.save('cdrs_'+ name, cdrs_dist_matrix)

    # slice cdr3s from dist_matrix, insert padding
    cdr3s_dist_matrix = dist_matrix[cdr3s, :][:, cdr3s]
    lngth = len(cdr3s_dist_matrix)
    cdr_len = len(cdr3l)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdr3s_dist_matrix)
    cdr3s_dist_matrix = numpy.pad(cdr3s_dist_matrix, [(0, 2*cdr3_PAD - lngth), (0, 2*cdr3_PAD - lngth)], mode='constant', constant_values=0)

    numpy.save('cdr3s_'+ name, cdr3s_dist_matrix)

    return cax

def animate_plot2 (i):
    pdb_code = pdbs[i].split(".")[0]
    pdb_filename = pdbs[i]
    name = '%s.npy' % (pdb_code)
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    # choose atomset for aLEF
    ## atomset choice
    #atoms = numpy.array(get_calphas(structure))
    #atoms = numpy.array(get_cbetas(structure))
    atoms = numpy.array(get_roids(structure))
    #atoms = numpy.array(get_scoids(structure))
    dist_matrix = calc_oid_dist_matrix(oid_one = atoms, oid_two = atoms, frame = i)
    lngth = len(dist_matrix)

    for i in range(len(lc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, hgth), 0),0)
        lngth, hgth = dist_matrix.shape

    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth), 0),1)
    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth + PAD-lc_len),0),0)

    lngth, hgth = dist_matrix.shape
    hc_anarci_shft = [x + PAD for x in hc_anarci['gaps']]

    for j in range(len(hc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1,lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1, hgth), 0),0)
        lngth = len(dist_matrix)

    lngth = len(dist_matrix)
    dm = numpy.pad(dist_matrix, [(0, pd2mx - lngth), (0, pd2mx - lngth)], mode='constant', constant_values=0)
    
    numpy.save(name, dm)
    contact_map = dist_matrix < 12.0
    cmap = 'plasma'
    cax = ax.imshow(dm
                    , cmap = plt.cm.get_cmap(cmap)
                    , interpolation = 'nearest', origin="lower")
    if i == 0:
        fig.colorbar(cax, cmap = plt.cm.get_cmap(cmap))

    # get cdr indices
    cdr1_PAD, cdr2_PAD, cdr3_PAD = 20, 12, 32
    cdrl1, cdrl2, cdrl3 = lc_anarci['cdr1'], lc_anarci['cdr2'], lc_anarci['cdr3']
    cdrh1, cdrh2, cdrh3 = hc_anarci['cdr1'], hc_anarci['cdr2'], hc_anarci['cdr3']

    #print(cdrl1)
    #print([lc_anarci['seq'][x] for x in cdrl1])
    #print(cdrl2)
    #print([lc_anarci['seq'][x] for x in cdrl2])
    #print(cdrl3)
    #print([lc_anarci['seq'][x] for x in cdrl3])
    #print(cdrh1)
    #print([hc_anarci['seq'][x] for x in cdrh1])
    #print(cdrh2)
    #print([hc_anarci['seq'][x] for x in cdrh2])
    #print(cdrh3)
    #print([hc_anarci['seq'][x] for x in cdrh3])
    
    cdrh1 = [x + PAD for x in cdrh1]
    cdrh2 = [x + PAD for x in cdrh2]
    cdrh3 = [x + PAD for x in cdrh3]

    cdrs = cdrl1 + cdrl2 + cdrl3 + cdrh1 + cdrh2 + cdrh3
    cdr3s = cdrl3 + cdrh3

    # slice cdrs from dist_matrix, insert padding
    cdrs_dist_matrix = dist_matrix[cdrs, :][:, cdrs]
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD +  cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)

    numpy.save('cdrs_'+ name, cdrs_dist_matrix)

    # slice cdr3s from dist_matrix, insert padding
    cdr3s_dist_matrix = dist_matrix[cdr3s, :][:, cdr3s]
    lngth = len(cdr3s_dist_matrix)
    cdr_len = len(cdr3l)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdr3s_dist_matrix)
    cdr3s_dist_matrix = numpy.pad(cdr3s_dist_matrix, [(0, 2*cdr3_PAD - lngth), (0, 2*cdr3_PAD - lngth)], mode='constant', constant_values=0)

    numpy.save('cdr3s_'+ name, cdr3s_dist_matrix)

    return cax

def animate_plot3 (i):
    pdb_code = pdbs[i].split(".")[0]
    pdb_filename = pdbs[i]
    name = '%s.npy' % (pdb_code)
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    # choose atomset for aLEF
    ## atomset choice
    #atoms = numpy.array(get_calphas(structure))
    #atoms = numpy.array(get_cbetas(structure))
    #atoms = numpy.array(get_roids(structure))
    atoms = numpy.array(get_scoids(structure))
    dist_matrix = calc_oid_dist_matrix(oid_one = atoms, oid_two = atoms, frame = i)
    lngth = len(dist_matrix)

    for i in range(len(lc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, lc_anarci['gaps'][i], numpy.full((1, hgth), 0),0)
        lngth, hgth = dist_matrix.shape

    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth), 0),1)
    dist_matrix = numpy.insert(dist_matrix, lc_len, numpy.full((PAD-lc_len, lngth + PAD-lc_len),0),0)

    lngth, hgth = dist_matrix.shape
    hc_anarci_shft = [x + PAD for x in hc_anarci['gaps']]

    for j in range(len(hc_anarci['gaps'])):
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1,lngth), 0),1)
        lngth, hgth = dist_matrix.shape
        dist_matrix = numpy.insert(dist_matrix, hc_anarci_shft[j], numpy.full((1, hgth), 0),0)
        lngth = len(dist_matrix)

    lngth = len(dist_matrix)
    dm = numpy.pad(dist_matrix, [(0, pd2mx - lngth), (0, pd2mx - lngth)], mode='constant', constant_values=0)
    
    numpy.save(name, dm)
    contact_map = dist_matrix < 12.0
    cmap = 'plasma'
    cax = ax.imshow(dm
                    , cmap = plt.cm.get_cmap(cmap)
                    , interpolation = 'nearest', origin="lower")
    if i == 0:
        fig.colorbar(cax, cmap = plt.cm.get_cmap(cmap))

    # get cdr indices
    cdr1_PAD, cdr2_PAD, cdr3_PAD = 20, 12, 32
    cdrl1, cdrl2, cdrl3 = lc_anarci['cdr1'], lc_anarci['cdr2'], lc_anarci['cdr3']
    cdrh1, cdrh2, cdrh3 = hc_anarci['cdr1'], hc_anarci['cdr2'], hc_anarci['cdr3']

    #print(cdrl1)
    #print([lc_anarci['seq'][x] for x in cdrl1])
    #print(cdrl2)
    #print([lc_anarci['seq'][x] for x in cdrl2])
    #print(cdrl3)
    #print([lc_anarci['seq'][x] for x in cdrl3])
    #print(cdrh1)
    #print([hc_anarci['seq'][x] for x in cdrh1])
    #print(cdrh2)
    #print([hc_anarci['seq'][x] for x in cdrh2])
    #print(cdrh3)
    #print([hc_anarci['seq'][x] for x in cdrh3])
    
    cdrh1 = [x + PAD for x in cdrh1]
    cdrh2 = [x + PAD for x in cdrh2]
    cdrh3 = [x + PAD for x in cdrh3]

    cdrs = cdrl1 + cdrl2 + cdrl3 + cdrh1 + cdrh2 + cdrh3
    cdr3s = cdrl3 + cdrh3

    # slice cdrs from dist_matrix, insert padding
    cdrs_dist_matrix = dist_matrix[cdrs, :][:, cdrs]
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrl3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr_len, numpy.full((cdr1_PAD-cdr_len, lngth + cdr1_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh2)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD +  cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr_len, numpy.full((cdr2_PAD-cdr_len, lngth + cdr2_PAD-cdr_len),0),0)
    lngth = len(cdrs_dist_matrix)
    cdr_len = len(cdrh3)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdrs_dist_matrix = numpy.insert(cdrs_dist_matrix, cdr1_PAD + cdr2_PAD + cdr3_PAD + cdr1_PAD + cdr2_PAD + cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)

    numpy.save('cdrs_'+ name, cdrs_dist_matrix)


    # slice cdr3s from dist_matrix, insert padding
    cdr3s_dist_matrix = dist_matrix[cdr3s, :][:, cdr3s]
    lngth = len(cdr3s_dist_matrix)
    cdr_len = len(cdr3l)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth), 0),1)
    cdr3s_dist_matrix = numpy.insert(cdr3s_dist_matrix, cdr_len, numpy.full((cdr3_PAD-cdr_len, lngth + cdr3_PAD-cdr_len),0),0)
    lngth = len(cdr3s_dist_matrix)
    cdr3s_dist_matrix = numpy.pad(cdr3s_dist_matrix, [(0, 2*cdr3_PAD - lngth), (0, 2*cdr3_PAD - lngth)], mode='constant', constant_values=0)

    numpy.save('cdr3s_'+ name, cdr3s_dist_matrix)st
    
    return cax

pd2mx = 320
PAD = 160
os.chdir('pathway/to/data/directory')
old_maps = ['CaDIST', 'CbDIST','roidDIST','scoidDIST']
maps = ['CaDIST', 'CbDIST','roidDIST','scoidDIST']
slices = ['cdrs', 'cdr3s']
#for slice in range(len(slices)):
for k in range(len(maps)):
    dirs = glob.glob('train/' + maps[k] + '/*/', recursive=True)
    cwd = os.getcwd()
    print(dirs)
    for i in range(len(dirs)):
        os.chdir(dirs[i])
        pdbs = glob.glob('*pdb')
        dirs_name = dirs[i].split(maps[k]+'/')[1]
        dirs_name = dirs_name.split('/')[0]
        print(dirs_name)
        fasta = glob.glob('*fasta')
        fasta = SeqIO.parse(open(fasta[0]), """fasta""")
        lc_len = None
        hc_len = None
        for chains in fasta:
            name, seq = chains.id, str(chains.seq)
            if "L" in name:
                lc_len = len(seq)
                anarci_L = [(name, seq)]
                results = anarci(anarci_L, scheme="imgt", output=False)
                numbering, alignment_details, hit_tables = results
                lc_anarci = [v for k, v in numbering[0][0][0]]
                lc_anarci_txt = ''.join(lc_anarci)
                lc_anarci_n = [k[0] for k, v in numbering[0][0][0]]
                gapl, cdr1l, cdr2l, cdr3l = [], [], [], []
                for i in range(0, len(lc_anarci)):
                    if lc_anarci_n[i] >= 27 and lc_anarci_n[i] <= 38:
                        cdr1l.append(i)
                    elif lc_anarci_n[i] >= 56 and lc_anarci_n[i] <= 65:
                        cdr2l.append(i)
                    elif lc_anarci_n[i] >= 105 and lc_anarci_n[i] <= 117:
                        cdr3l.append(i)
                for i in range(0, len(lc_anarci)):
                    if lc_anarci[i] == '-':
                        gapl.append(i)


                lc_anarci = {'seq': lc_anarci, 'anarci_n': lc_anarci_n, 'gaps': gapl, 'cdr1': cdr1l, 'cdr2': cdr2l, 'cdr3': cdr3l}
                lc_len = len(lc_anarci_txt)
                lc_anarci_txt = lc_anarci_txt.ljust(PAD,'-')
                lc_len = lc_len 
                with open('L_%s.txt' % dirs_name, "w") as text_file:
                    text_file.write(lc_anarci_txt)
                with open('L_%s.json' % dirs_name, "w") as json_file:
                    json.dump(lc_anarci, json_file)
                #seq = numpy.fromstring(seq, dtype=str)
                #numpy.savetxt('L_%s.txt' % dirs_name, seq) 
                continue
            elif "H" in name:
                hc_len = len(seq)
                anarci_H = [(name, seq)]
                results = anarci(anarci_H, scheme="imgt", output=False)
                numbering, alignment_details, hit_tables = results
                hc_anarci = [v for k, v in numbering[0][0][0]]
                hc_anarci_txt = ''.join(hc_anarci)
                hc_anarci_n = [k[0] for k, v in numbering[0][0][0]]
                gaph, cdr1h, cdr2h, cdr3h = [], [], [], []
                for i in range(0, len(hc_anarci)):
                    if hc_anarci_n[i] >= 27 and hc_anarci_n[i] <= 38:
                        cdr1h.append(i)
                    elif hc_anarci_n[i] >= 56 and hc_anarci_n[i] <= 65:
                        cdr2h.append(i)
                    elif hc_anarci_n[i] >= 105 and hc_anarci_n[i] <= 117:
                        cdr3h.append(i)

                for i in range(0, len(hc_anarci)):
                    if hc_anarci[i] == '-':
                        gaph.append(i)
            
                hc_anarci = {'seq': hc_anarci, 'anarci_n': hc_anarci_n, 'gaps': gaph, 'cdr1': cdr1h, 'cdr2': cdr2h, 'cdr3': cdr3h}

                hc_len = len(hc_anarci_txt)
                

                hc_anarci_txt = hc_anarci_txt.ljust(PAD,'-')
                hc_len = len(hc_anarci_txt)
                with open('H_%s.txt' % dirs_name, "w") as text_file:
                    text_file.write(hc_anarci_txt)
                with open('H_%s.json' % dirs_name, "w") as json_file:
                    json.dump(hc_anarci, json_file) 
                #seq = numpy.fromstring(seq, dtype=str)
                #numpy.savetxt('H_%s.txt' % dirs_name, seq)
                continue
            else:
                print('fasta parser appends 2 chains per Fv (assumes H or L in chain name)')
                break

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if maps[k] == 'CaDIST':
            ani = FuncAnimation(fig,
                                animate_plot,
                                frames=len(pdbs),
                                init_func=init,
                                interval=100)

        elif maps[k] == 'CbDIST':
            ani = FuncAnimation(fig,
                                animate_plot1,
                                frames=len(pdbs),
                                init_func=init,
                                interval=100)
            
        elif maps[k] == 'roidDIST':
            ani = FuncAnimation(fig,
                                animate_plot2,
                                frames=len(pdbs),
                                init_func=init,
                                interval=100)
        elif maps[k] == 'scoidDIST':
            ani = FuncAnimation(fig,
                                animate_plot3,
                                frames=len(pdbs),
                                init_func=init,
                                interval=100)
        else:
            print('can not find interaction map')
        
        ani.save('%s.gif' % dirs_name, fps=10)
        plt.show()
        npys = glob.glob('DAB*.npy') #glob ensemble dist maps
        hr = numpy.empty((pd2mx,pd2mx)) #instaniate numpy nd array for stacking
        for j in range(len(npys)):
            print(npys[j])
            npy = numpy.load(npys[j])
            hr = numpy.dstack((hr, npy)) #stack ensemble dist maps
        hr = numpy.delete(hr, 0, axis = -1) #delete instantiation [:, :, 0]
        hr = numpy.average(hr, axis = -1) #average ensemble stack
        numpy.save('HR.npy', hr) #call this one high resolution
        os.chdir(cwd)