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
Helpers for parsing protein structure files and generating contact maps.
"""

import gzip
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import pairwise_distances
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.PDB.Polypeptide import three_to_one, is_aa


def idx_align(pdb_seq: str, og_seq: str):
    """Maps indices of original and PDB sequences to mutual alignment
    Args:
        pdb_seq: Sequence of amino acids as extracted from PDB file
        og_seq: Original sequence of amino acids
    Returns:
        Tuple where the first element is a list of integers containing the indices in
        the PDB sequence for which aligned information is available and the second element
        is a list of integers containing the indices in the original sequence
        for which aligned information is available
    """
    alignment = pairwise2.align.globalxx(Seq(og_seq), Seq(pdb_seq))[0]
    og_seq_align = alignment.seqA
    og_idxs = []
    og_offset = 0

    pdb_seq_align = alignment.seqB
    pdb_idxs = []
    pdb_offset = 0
    for i in range(len(pdb_seq_align)):
        valid = True
        pdb_res = pdb_seq_align[i]
        og_res = og_seq_align[i]
        if pdb_res == "-":
            pdb_offset += 1
            valid = False
        if og_res == "-":
            og_offset += 1
            valid = False
        if valid and (pdb_res == og_res):
            pdb_idxs.append(i - pdb_offset)
            og_idxs.append(i - og_offset)

    return pdb_idxs, og_idxs


def calc_adj_matrix(structure, dist_thresh=10.0):
    """Returns an adjacency matrix for a PDB structure
    Args:
        structure : BioPython PDB Structure object representing a protein containing N residues
        dist_thresh : float, optional. Value specifying threshold for classifying contacts between residues

    Returns:
        Tuple where the first element is a np.ndarray adjacency matrix with shape NxN, where N is the number of residues in `structure` and the second element is np.ndarray NxN matrix specifying distances between residues.
    """
    residues = list(structure.get_residues())
    coords = np.asarray([residue["CA"].coord for residue in residues])
    dist_mat = pairwise_distances(coords, metric="euclidean")
    return 1 * (dist_mat < dist_thresh), dist_mat


def gunzip_to_ram(gzip_file_path):
    """
    gunzip a gzip file and decode it to a io.StringIO object.

    Args:
        gzip_file_path: String. Gunzip filepath.

    Returns:
        io.StringIO object.
    """
    content = []
    with gzip.open(gzip_file_path, "rb") as f:
        for line in f:
            content.append(line.decode("utf-8"))

    temp_fp = StringIO("".join(content))
    return temp_fp


def _parse_structure(parser, name, file_path):
    """Parse a .pdb or .cif file into a structure object.
    The file can be gzipped.

    Args:
        parser: a Bio.PDB.PDBParser or Bio.PDB.MMCIFParser instance.
        name: String. name of protein
        file_path: String. Filpath of the pdb or cif file to be read.

    Retruns:
        a Bio.PDB.Structure object representing the protein structure.

    """
    if pd.isnull(file_path):
        return None
    if file_path.endswith(".gz"):
        structure = parser.get_structure(name, gunzip_to_ram(file_path))
    else:  # not gzipped
        structure = parser.get_structure(name, file_path)
    return structure


parse_pdb_structure = _parse_structure  # for backward compatiblity


def parse_structure(pdb_parser, cif_parser, name, file_path):
    """Parse a .pdb file or .cif file into a structure object.
    The file can be gzipped.

    Args:
        pdb_parser: a Bio.PDB.PDBParser instance
        cif_parser: Bio.PDB.MMCIFParser instance
        name: String. name of protein
        file_path: String. Filpath of the pdb or cif file to be read.

    Return:
        a Bio.PDB.Structure object representing the protein structure.
    """
    if file_path.rstrip(".gz").endswith("pdb"):
        return _parse_structure(pdb_parser, name, file_path)
    else:
        return _parse_structure(cif_parser, name, file_path)


def get_energy(pdb_file_path):
    """Get total pose energy from a PDB file.

    Args:
        pdb_file_path: String. Path to the pdb file.

    Return:
        Energy score.

    """
    if pdb_file_path.endswith(".pdb.gz"):
        pdb_file = gunzip_to_ram(pdb_file_path)
    else:
        pdb_file = open(pdb_file_path, "r")

    parse = False
    score = None
    for line in pdb_file:
        if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
            parse = True
        if parse and line.startswith("pose"):
            score = float(line.strip().split()[-1])
    return score


def three_to_one_standard(res):
    """Encode non-standard AA to X.

    Args:
        res: a Bio.PDB.Residue object representing the residue.

    Return:
        String. One letter code of the residue.
    """
    if not is_aa(res, standard=True):
        return "X"
    return three_to_one(res)


def is_aa_by_target_atoms(res):
    """Tell if a Residue object is AA

    Args:
        res: a Bio.PDB.Residue object representing the residue.

    Return:
       Bool. Wheather or not the residue is AA.
    """
    target_atoms = ["N", "CA", "C", "O"]
    for atom in target_atoms:
        try:
            res[atom]
        except KeyError:
            return False
    return True