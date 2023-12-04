#!/usr/bin/env python3
#    Biomol2Clust: Tool to cluster docked poses, structural states of ligands/peptides, 
#    and conformational variants of biological macromolecules using state-of-the-art machine learning
#
#    Copyright (C) 2020-2021 Timonina D., Sharapova Y., Švedas V., Suplatov D.
#    "Dmitry Suplatov" <d.a.suplatov@belozersky.msu.ru>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import rmsdm
import sklearn.cluster
import hdbscan
import math
from collections import Counter
import os
from os import listdir
from os.path import isfile, join
from tabulate import tabulate
import sys
import multiprocessing
import datetime
import re
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

class Atom:
    def __init__(self, name, coords):
        self.name = name
        self.coords = coords

class Molecule:
    def __init__(self):
        self.atoms = []
        self.list_of_all_coords = []
        self.list_of_noh_coords = []
        self.name = None
    def add_atom(self, atom):
        self.atoms.append(atom)
        self.list_of_all_coords.append(atom.coords)
        if not re.match(r'^[0-9]*H', atom.name):
            self.list_of_noh_coords.append(atom.coords)
    def set_energy(self, energy):
        self.dg = energy

def dist_matr_maker(noh, align, all_molecules, i):
    arr = []
    for _ in range(i):
        arr.append(0)
    if noh:
        if align:
            for j in range(i, len(all_molecules)):
                arr.append(rmsdm.kabsch_rmsd(np.array(all_molecules[i].list_of_noh_coords), np.array(all_molecules[j].list_of_noh_coords), translate=True))
        else:
            for j in range(i, len(all_molecules)):
                arr.append(rmsdm.rmsd(all_molecules[i].list_of_noh_coords, all_molecules[j].list_of_noh_coords))
    else:
        if align:
            for j in range(i, len(all_molecules)):
                arr.append(rmsdm.kabsch_rmsd(np.array(all_molecules[i].list_of_all_coords), np.array(all_molecules[j].list_of_all_coords), translate=True))
        else:
            for j in range(i,len(all_molecules)):
                arr.append(rmsdm.rmsd(all_molecules[i].list_of_all_coords, all_molecules[j].list_of_all_coords))
    return arr

def name_of_aa(line):
    return line[16:20].replace(' ', '')

def name_of_at(line):
    return(line[12:16].replace(' ',''))

def reading_folder_pdbs(folder):
    all_molecules = []
    atom_seq = []
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for file in files:
        assert file.endswith('.pdb'), "All files in folder should be in pdb-format"
        molecule = Molecule()
        molecule.filename = file
        molecule.full_text = ''
        atom_seq.append([])
        with open(os.path.join(folder, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("ATOM") and line.find(" H") == -1:
                    molecule.full_text += line
                    coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    atom_seq[-1].append(name_of_at(line))
                    name = name_of_at(line)
                    atom = Atom(name, coords)
                    molecule.add_atom(atom)
        molecule.name = file
        all_molecules.append(molecule)
    lens_of_mols = [len(mol.atoms) for mol in all_molecules]
    assert all(x == lens_of_mols[0] for x in lens_of_mols), "Molecular lengths shoud be the same"
    if not np.all(np.array(atom_seq) == atom_seq[0]):
        print('Warning: Atoms are not in the same order for all molecules')
    return all_molecules

def reading_sdf_file(filename):
    all_molecules = []
    atom_seq = []
    molid = 0
    with open(filename, 'r') as f:
        splited_text = f.read().split('$$$$')[0:-1]
    for mol in splited_text:
        molid += 1
        molecule = Molecule()
        molecule.full_text = mol.strip()
        molecule.name = str(molid)
        lines = mol.split('\n')
        atom_seq.append([])
        for line in lines:
            if len(line) > 25 and line[5] == '.' and line[15] == '.' and line[25] == '.':
                line_splt = line.split()
                coords = [float(line_splt[0]), float(line_splt[1]), float(line_splt[2])]
                atom_seq[-1].append(line_splt[3])
                name = line_splt[3]
                atom = Atom(name, coords)
                molecule.add_atom(atom)
            if line.startswith('M') and 'dG=' in line:
                line_splt = np.array(line.split())
                all_indices = np.where(line_splt == 'dG=')[0] + 1
                all_energies = line_splt[all_indices[all_indices < len(line_splt)]].astype('float64')
                assert np.all(all_energies == all_energies[0]), 'Error: Multiple dG statements in one line at pose #{pose}: \n{line}'.format(pose=molid, line=line)
                if hasattr(molecule, 'dg') and molecule.dg != all_energies[0]:
                    print ('Error: Multiple dG statements for pose #{pose}: {dg1}, {dg2}'.format(pose=molid, dg1=molecule.dg, dg2=all_energies[0]))
                    exit(1)
                molecule.set_energy(all_energies[0])
        if not hasattr(molecule, 'dg'):
            molecule.set_energy(0)
            print('Warning: No dG data was provided for pose #{} (dG set to 0)'.format(molid))
        all_molecules.append(molecule)      
    if not np.all(np.array(atom_seq) == atom_seq[0]):
        print('Warning: Atoms are not in the same order for all molecules')
    return all_molecules

def translate_rotation_new_coords(current, centr):
    current = np.array(current)
    centr = np.array(centr)
    current_cent = current - rmsdm.centroid(current)
    centr_cent = centr - rmsdm.centroid(centr)
    U = rmsdm.kabsch(current_cent, centr_cent)
    current_cent = np.dot(current_cent, U) 
    current_cent += rmsdm.centroid(centr)
    return current_cent
    
def print_cluster_sdf(dist_matr, align, output, rank, lab, mols, size, mean, best_energy):
    if lab == -1: 
        file_name = "outliers_size{}_meandg={}_bestdg={}.sdf".format(size, mean, best_energy)
    else: 
        file_name = "cluster{}_size{}_meandg={}_bestdg={}.sdf".format(rank, size, mean, best_energy)

    num_centr_mol = np.argmin(np.sum(dist_matr, axis=0))
    with open(os.path.join(output, "REPCONF_" + file_name), 'w') as f:
        f.write(mols[num_centr_mol].full_text)
        f.write('\n$$$$\n')
    if align:
        with open(os.path.join(output, file_name), 'w') as f:
            for mol in sorted(mols, key=lambda x: x.dg):
                flag = 0
                new_coords = translate_rotation_new_coords(mol.list_of_all_coords, mols[num_centr_mol].list_of_all_coords)    
                for line in mol.full_text.splitlines():
                    if len(line) > 25 and line[5] == '.' and line[15] == '.' and line[25] == '.':   
                        line = list(line)
                        line[0:30] = ' '*30
                        line[10 - len("{:.4f}".format(new_coords[flag][0])):10] = "{:.4f}".format(new_coords[flag][0])
                        line[20 - len("{:.4f}".format(new_coords[flag][1])):20] = "{:.4f}".format(new_coords[flag][1])
                        line[30 - len("{:.4f}".format(new_coords[flag][2])):30] = "{:.4f}".format(new_coords[flag][2])
                        s = ''.join(line)
                        f.write(s + '\n')    
                        flag += 1
                    else:
                        f.write(line + '\n')
                f.write('$$$$\n')    
            
    else:            
        with open(os.path.join(output, file_name), 'w') as f:
            for mol in sorted(mols, key= lambda x: x.dg):
                f.write(mol.full_text)
                f.write('\n$$$$\n')

def print_cluster_pdb(dist_matr, align, output, rank, lab, mols, size):
    if lab == -1: 
        file_name = "outliers_size{}.pdb".format(size)
    else: 
        file_name = "cluster{}_size{}.pdb".format(rank, size)
    num_centr_mol = np.argmin(np.sum(dist_matr, axis=0))
    with open(os.path.join(output, "REPCONF_" + file_name), 'w') as f:
        f.write(mols[num_centr_mol].full_text)
    if align:
        with open(os.path.join(output, file_name), 'w') as f:
            for i, mol in enumerate(mols):
                new_coords = translate_rotation_new_coords(mol.list_of_all_coords, mols[num_centr_mol].list_of_all_coords)
                f.write('MODEL {}\n'.format(i + 1))
                for j, line in enumerate(mol.full_text.splitlines()):
                    line = list(line)
                    line[30:54] = ' '*24
                    line[38 - len("{:.3f}".format(new_coords[j][0])):38] = "{:.3f}".format(new_coords[j][0])
                    line[46 - len("{:.3f}".format(new_coords[j][1])):46] = "{:.3f}".format(new_coords[j][1])
                    line[54 - len("{:.3f}".format(new_coords[j][2])):54] = "{:.3f}".format(new_coords[j][2])
                    s = ''.join(line)
                    f.write(s + '\n')
                f.write('TER\nENDMDL\n')
            f.write('END') 
    else:
        with open(os.path.join(output, file_name), 'w') as f:
            for i, mol in enumerate(mols):
                f.write('MODEL {}\n'.format(i + 1))
                f.write(mol.full_text)
                f.write('TER\nENDMDL\n')
            f.write('END')  
            
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def print_result_files(output, dict_of_counts, sorted_labs, labels, all_molecules, input_type, dist_matr, align):
    table = [["Rank", "Сardinality", "Mean(energy)", "Std(energy)", "Best(energy)", "RMSD(mean)", "RMSD(stdev)"]]
    rank = 1
    for lab in sorted_labs:
        dist_matr_for_this_lab = dist_matr[labels == lab].T[labels == lab]
        if dict_of_counts[lab] > 1:
            mean_rmsd = round(upper_tri_masking(dist_matr_for_this_lab).mean(), 3)
            if dict_of_counts[lab] > 5:
                std_rmsd = round(upper_tri_masking(dist_matr_for_this_lab).std(), 3)
            else:
                std_rmsd = 'N/A'
        else:
            mean_rmsd = 'N/A'
            std_rmsd = 'N/A'
        
        if input_type == 'sdf':
            mean_energy = round(np.mean([i.dg for i in all_molecules[labels == lab]]), 3)
            best_energy = round(min([i.dg for i in all_molecules[labels == lab]]), 3)
            if dict_of_counts[lab] > 5:
                std_energy = round(np.std([i.dg for i in all_molecules[labels == lab]]), 3)
            else:
                std_energy = 'N/A'
            print_cluster_sdf(dist_matr_for_this_lab, align, output, rank, lab, all_molecules[labels == lab], dict_of_counts[lab], mean_energy, best_energy)
        elif input_type == 'pdb':
            mean_energy = 'N/A'
            best_energy = 'N/A'
            std_energy = 'N/A'  
            print_cluster_pdb(dist_matr_for_this_lab, align, output, rank, lab, all_molecules[labels == lab], dict_of_counts[lab])
        if lab == -1:
            row_outliers = ['outliers', dict_of_counts[lab], mean_energy, std_energy, best_energy, 'N/A', 'N/A']
        else:
            table.append([rank, dict_of_counts[lab], mean_energy, std_energy, best_energy, mean_rmsd, std_rmsd])
        if lab != -1:
            rank += 1
    if -1 in labels:
        table.append(row_outliers)
    with open(os.path.join(output, 'RESULTS.txt'), 'w') as f:
        f.write(tabulate(table)+"\n")
    with open(os.path.join(output, 'RESULTS_cluster_content_names.txt'), 'w') as f:
        rank = 1
        for lab in sorted_labs:
            if lab == -1:
                continue
            if input_type == 'sdf':
                mols = sorted(all_molecules[labels == lab], key= lambda x: x.dg)
            else:
                mols = all_molecules[labels == lab]
            f.write('Structures in cluster #{}:\n'.format(rank))
            for mol in mols:
                f.write('{} '.format(mol.name))
            f.write('\n')
            rank += 1
        if -1 in labels:
            f.write('Outliers:\n')
            if input_type == 'sdf':
                mols = sorted(all_molecules[labels == -1], key= lambda x: x.dg)
            else:
                mols = all_molecules[labels == -1]
            for mol in mols:
                f.write('{} '.format(mol.name))


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    dict_of_sys_argv = dict([arg.split('=') for arg in sys.argv[1:]])
    
    if len(dict_of_sys_argv) == 0:
        print('''
                     :-) Biomol2Clust (-:
      
 Tool to cluster docked poses, structural states of ligands/peptides, 
     and conformational variants of biological macromolecules 
             using state-of-the-art machine learning

                  :-) v. 1.3 2021-Jan-29 (-:

Usage:   python3 main.py input=</path/to/file> output=</path/to/folder> [options]
Example: python3 main.py input=./ligands.sdf output=./results
         python3 main.py input=./folder_with_pdbs/ output=./results

Mandatory input parameters:
===========================
input=<string>         # Path to input sdf-file or folder with pdb-files
output=<string>        # Path to folder to store results

Utilization of computing resources: 
===================================
cpu_threads=<int>      # Number of parallel CPU threads to utilize by OPTICS or DBSCAN (the default is "all" physically available)
                       # HDBSCAN will utilize only one thread/core

Cluster analysis methods:
=========================
align=[true|false]          # If true, each pair of input conformations is superimposed for best-fit prior to calculating pariwise RMSD (default=false)
noh=[true|false]            # If true, hydrogen atoms are dismissed when calculating pariwise RMSD similarity matrix (default=false)
method=hdbscan              # Use HDBSCAN automatic method (default) 
method=optics               # Use OPTICS automatic method
method=dbscan eps=<float>   # Use DBSCAN method for manual fine-tuning of the results by specifying the ‘eps’ value (eps > 0)

Cluster size:
=========================
min_samples=<int>           # The 'min_samples' parameter of HDBSCAN, OPTICS, and DBSCAN that regulates the number of points in a neighborhood 
                              for a point to be considered as the cluster core (default=Null for HDBSCAN; 5 for OPTICS and DBSCAN)
min_cluster_size=<int>      # The HDBSCAN 'min_cluster_size' parameter to regulate the minimal size of a cluster (default=5)

Documentation:
==============
The latest version of the program, documentation and examples are available open-access at https://biokinet.belozersky.msu.ru/Biomol2Clust

''')
        quit()

    if 'input' in dict_of_sys_argv:
        input = os.path.abspath(dict_of_sys_argv['input'])
        if os.path.isfile(input): 
            assert input.endswith('.sdf'), 'File should have .sdf format'
            input_type = 'sdf'
        elif os.path.isdir(input):
            input_type = 'pdb'
        else:
            raise Exception('Something wrong with input. There is no folder or file {}'.format(input))
    else:
        raise Exception('Error: No input file or folder')

    if 'cpu_threads' in dict_of_sys_argv:
        cpu_threads = int(dict_of_sys_argv['cpu_threads'])
    else:
        cpu_threads = multiprocessing.cpu_count()
    
    if 'align' in dict_of_sys_argv:
        align = dict_of_sys_argv['align'] == 'true'
    else:
        align = False

    if 'noh' in dict_of_sys_argv:
        noh = dict_of_sys_argv['noh'] == 'true'
    else:
        noh = False

    if 'method' in dict_of_sys_argv:
        if dict_of_sys_argv['method'] not in ['optics', 'dbscan', 'hdbscan']: raise Exception('Method must be optics, dbscan or hdbscan')
        method_of_clustering = dict_of_sys_argv['method']
    else:
         method_of_clustering = 'hdbscan'

    if 'eps' in dict_of_sys_argv:
        eps = float(dict_of_sys_argv['eps'])
    elif method_of_clustering == 'dbscan':
            raise Exception('If you use dbscan-method, you should choose eps')

    if 'output' in dict_of_sys_argv:
        output = os.path.abspath(dict_of_sys_argv['output'])
    else:
        raise Exception('Choose folder for output')

    if os.path.exists(output) and os.path.isdir(output):
        if os.listdir(output):
            raise Exception('Error: Folder {} exists and is not empty'.format(output))
    else:
        os.makedirs(output)

    if 'min_samples' in dict_of_sys_argv:
        min_samples = int(dict_of_sys_argv['min_samples'])
    else:
        if method_of_clustering == 'hdbscan':
            min_samples = None
        else:
            min_samples = 5
    
    if 'min_cluster_size' in dict_of_sys_argv:
        min_cluster_size = int(dict_of_sys_argv['min_cluster_size'])
    else:
        min_cluster_size = 5
    
    print('Info: Started at', str(time_start).split('.')[0])
    print('Info: Input mode: {} align={} noh={} {}'.format(input_type.upper(), str(align).lower(), str(noh).lower(), input))
    
    if input_type == 'sdf':
        all_molecules = np.array(reading_sdf_file(input))
    elif input_type == 'pdb': 
        all_molecules = np.array(reading_folder_pdbs(input))
    
    print('Info: Number of conformations: {}'.format(len(all_molecules)))
    print('Info: Calculating pairwise RMSD similarity matrix')

    dist_matr = []

    with Pool(cpu_threads) as p:
        arg = range(len(all_molecules))
        func = partial(dist_matr_maker, noh, align, all_molecules)
        res = list(tqdm(p.imap(func, arg), total=len(all_molecules)))
    p.join()

    for j in res:
        dist_matr.append(j)
    dist_matr = np.asarray(dist_matr)
    print(dist_matr.shape)
    print(len(all_molecules))
    for i in range(len(all_molecules)):
        for j in range(i,len(all_molecules)):
            dist_matr[j][i] = dist_matr[i][j]
    
    if method_of_clustering == 'hdbscan':
        print('\nInfo: Running cluster analysis using HDBSCAN (min_sample={}; min_cluster_size={})'.format(min_samples, min_cluster_size))
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=eps)
    elif method_of_clustering == 'optics':
        print('\nInfo: Running cluster analysis using OPTICS (min_sample={})'.format(min_samples))
        clusterer = sklearn.cluster.OPTICS(metric='precomputed', n_jobs=cpu_threads, min_samples=min_samples)
    elif method_of_clustering == 'dbscan':
        print('\nInfo: Running cluster analysis using DBSCAN (min_sample={}; eps={})'.format(min_samples, eps))
        clusterer = sklearn.cluster.DBSCAN(metric='precomputed', n_jobs=cpu_threads, eps=eps, min_samples=min_samples)

    res = clusterer.fit(dist_matr)

    labels = res.labels_
    print('Info: Number of clusters: {} ({} outliers)'.format(len(set(labels)) - (list(labels).count(-1) > 0),list(labels).count(-1)))
    dict_of_counts = Counter(labels)
    sorted_labs = sorted(dict_of_counts, key=dict_of_counts.get, reverse=True)
    
    print_result_files(output, dict_of_counts, sorted_labs, labels, all_molecules, input_type, dist_matr, align)
    print('Info: Printed results to {}'.format(output))
    time_end = datetime.datetime.now()
    print('Info: Ended at', str(time_end).split('.')[0])
