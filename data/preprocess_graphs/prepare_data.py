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



"""
Parse PDB files to extract the coordinates of the 4 key atoms from AAs to
generate json records compatible to the GNN models 

This is run before graph_structs.py

"""

import json
import os
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import glob
from Bio import SeqIO
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

import xpdb
from contact_map_utils import parse_pdb_structure


def parse_args():
    """Prepare argument parser.

    Args:

    Return:

    """
    parser = argparse.ArgumentParser(
        description="Generate GVP ProteinGraph datasets representing protein"
        + "structures."
    )
    parser.add_argument(
        "--data-file",
        help="Path to protein data frame, including sequences, paths to"
        + " structure files and labels",
        required=False,
        default = "fv_data.csv"
    )
    parser.add_argument(
        "-t",
        "--target-variable",
        help="target variable in the protein data frame",
        required=False,
        default='target'
    )
    parser.add_argument("-o", "--output", help="output dir for graphs")

    args = parser.parse_args()
    return args


def get_atom_coords(residue, target_atoms=["N", "CA", "C", "O"]):
    """Extract the coordinates of the target_atoms from an AA residue.

    Args:
        residue: a Bio.PDB.Residue object representing the residue.
        target_atoms: Target atoms which residues will be returned.

    Retruns:
        Array of residue's target atoms (in the same order as target atoms).
    """
    return np.asarray([residue[atom].coord for atom in target_atoms])


def structure_to_coords(struct, target_atoms=["N", "CA", "C", "O"], name=""):
    """Convert a PDB structure in to coordinates of target atoms from all AAs

    Args:
        struct: a Bio.PDB.Structure object representing the protein structure
        target_atoms: Target atoms which residues will be returned.
        name: String. Name of the structure

    Return:
        Dictionary with the pdb sequence, atom 3D coordinates and name.
    """
    output = {}
    # get AA sequence in the pdb structure
    pdb_seq = "".join(
        [three_to_one(res.get_resname()) for res in struct.get_residues() if res.get_resname() != "NME"]
    )
    output["seq"] = pdb_seq
    # get the atom coords
    coords = np.asarray(
        [
            get_atom_coords(res, target_atoms=target_atoms)
            for res in struct.get_residues() if res.get_resname() != "NME"
        ]
    )
    output["coords"] = coords.tolist()
    output["name"] = name
    return output


def parse_pdb_gz_to_json_record(parser, sequence, pdb_file_path, name=""):
    """
    Reads and reformats a pdb strcuture into a dictionary.

    Args:
        parser: a Bio.PDB.PDBParser or Bio.PDB.MMCIFParser instance.
        sequence: String. Sequence of the structure.
        pdb_file_path: String. Path to the pdb file.
        name: String. Name of the protein.

    Return:
        Dictionary with the pdb sequence, atom 3D coordinates and name.
    """
    struct = parse_pdb_structure(parser, sequence, pdb_file_path)
    record = structure_to_coords(struct, name=name)
    return record


def main():
    """
    Data preparation main script: Load data, parses PDB, processes structures, segregate records and write to disk. Configuration via commandline arguments.

    Args:

    Return:

    """
    args = parse_args()

    #if args.data_file exists
    if not os.path.exists(args.data_file):
        # read through train and holdout directories and create pandas dataframe compatible with lm-gvp preprocessor
        data_split = ["train", "holdout"]
        data = ["primary", "structure_path","target", "split"]
        #data frame
        df = pd.DataFrame(columns=data)
        os.chdir('/pathway/to/data/directory/')
        for split in data_split:
            dirs = glob.glob(split + '/' + 'roidDIST' + '/*/', recursive=True)
            cwd = os.getcwd()
            #print(dirs)
            for i in range(len(dirs)):
                os.chdir(dirs[i])
                fasta = glob.glob('*fasta')
                fasta = SeqIO.parse(open(fasta[0]), """fasta""")
                for seq in fasta:
                    name, sequence = seq.id, str(seq.seq)
                    if 'L' in name:
                        l_seq = sequence
                    elif 'H' in name:
                        h_seq = sequence
                seq = l_seq + h_seq
                # read prop.npy
                prop = float(np.load('prop.npy'))
                pdb = glob.glob('*_01.pdb')
                path_to_pdb = os.path.join(dirs[i], pdb[0])
                path_to_pdb = '/pathway/to/data/directory/' + path_to_pdb
                df = df.append({'primary': seq, 
                                'structure_path': path_to_pdb,
                                'target': prop, 
                                'split': split}, ignore_index=True)
                os.chdir(cwd)

        df.to_csv('preprocess_graphs/fv_data.csv')
    else:
        df = pd.read_csv(args.data_file)

    # load data into lm-gvp preprocessor

    # PDB parser
    sloppyparser = PDBParser(
        QUIET=True,
        PERMISSIVE=True,
        structure_builder=xpdb.SloppyStructureBuilder(),
    )

    # 2. Parallel parsing structures and converting to protein records
    records = Parallel(n_jobs=-1)(
        delayed(parse_pdb_gz_to_json_record)(
            sloppyparser,
            df.iloc[i]["primary"],
            df.iloc[i]["structure_path"],
            df.iloc[i]["structure_path"].split("/")[-1],
        )
        for i in tqdm(range(df.shape[0]))
    )

    # 3. Segregate records by splits
    splitted_records = defaultdict(list)
    for i, rec in enumerate(records):
        row = df.iloc[i]
        target = row[args.target_variable]
        split = row["split"]
        rec["target"] = target
        splitted_records[split].append(rec)

    # 4. write to disk
    for split, records in splitted_records.items():
        print(split, "number of proteins:", len(records))
        outfile = os.path.join(args.output, f"proteins_{split}.json")
        json.dump(records, open(outfile, "w"))

    # iterate through lm-gvp preprocesed json; create torch_geometric object for each Fv and save to disk
    
    return None


if __name__ == "__main__":
    main()