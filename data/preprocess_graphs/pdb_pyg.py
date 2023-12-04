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

import math
import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import re
import os 
import json

def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.

    Args:
        tensor: Torch tensor to be normalized.
        dim: Integer. Dimension to normalize across.

    Returns:
        Normalized tensor with zeros instead of nan's or infinity values.

    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )

def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].


    Args:
        D: generic torch tensor
        D_min: Float. Minimum of the sequence of numbers created.
        D_max: Float. Max of the sequence of numbers created.
        D_count: Positive integer. Count of the numbers in the sequence. It is also lenght of the new dimension (-1) created in D.
        device: Device where D is stored.

    Return:
        Input `D` matrix with an RBF embedding along axis -1.
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF

def prep_seq(seq):
    """
    Adding spaces between AAs and replace rare AA [UZOB] to X.
    ref: https://huggingface.co/Rostlab/prot_bert.

    Args
        seq: a string of AA sequence.

    Returns:
        String representing the input sequence where U,Z,O and B has been replaced by X.
    """
    seq_spaced = " ".join(seq)
    seq_input = re.sub(r"[UZOB]", "X", seq_spaced)
    return seq_input

def get_dataset(dataset = None, split="train", num_outputs = 1):
    """Load data from files, then transform into appropriate
    Dataset objects.
    Args:
        task: one of ['cc', 'bp', 'mf', 'protease', 'flu']
        model_type: one of ['seq', 'struct', 'seq_struct']
        split: one of ['train', 'valid', 'test']

    Return:
        Torch dataset.
    """
    
    num_outputs = num_outputs
    pos_weights = None
    #dataset = ProteinGraphDatasetWithTarget(dataset, preprocess=False)
    dataset = BertProteinGraphDatasetWithTarget(
                dataset, preprocess=False
            )
    dataset.num_outputs = num_outputs
    dataset.pos_weights = pos_weights
    return dataset

class BaseProteinGraphDataset(data.Dataset):
    """Dataset for the Base Protein Graph."""

    def __init__(
        self,
        data_list,
        num_positional_embeddings=16,
        top_k=30,
        num_rbf=16,
        device="cpu",
        preprocess=True,
    ):
        """
        Initializes the dataset

        Args:
            data_list: List containint the initial dataset
            num_positional_embeddings: Integer specifying the number of positional embeddings.
            top_k: Integer k to use in kNN when constructing the graph
            num_rbf: Integer specifying number of radial basis functions
            device: Device to allocate the tensors.
            preprocess: Whether to preprocess the data_list.

        Returns:
            None
        """

        super(BaseProteinGraphDataset, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        if preprocess:
            print("Preprocessing data...")
            self._preprocess()

    def _preprocess(self):

        """Preprocess all the records in `data_list` with
        `_featurize_as_graph`"""
        for i in tqdm.tqdm(range(len(self.data_list))):
            self.data_list[i] = self._featurize_as_graph(self.data_list[i])

    def __getitem__(self, i):
        if isinstance(self.data_list[i], dict):
            # if not processed, process this instance and update
            self.data_list[i] = self._featurize_as_graph(self.data_list[i])
        return self.data_list[i]

    def __len__(self):
        return len(self.data_list)

    def _featurize_as_graph(self, protein):
        """Placeholder for the _featurize_as_graph method implemented in child classes.

        Args:
            protein: a dict representing a data object

        Returns:
            None
        """
        raise NotImplementedError

    def _dihedrals(self, X, eps=1e-7):
        """Compute sines and cosines dihedral angles (phi, psi, and omega)

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]
            eps: Float defining the epsilon using to clamp the angle between normals: min= -1*eps, max=1-eps

        Returns:
            Sines and cosines dihedral angles as a torch.Tensor of shape [seq_len, 6]
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index, num_embeddings=None):
        """Creates and returns the positional embeddings.

        Args:
            edge_index: torch.Tensor representing edges in COO format with shape [2, num_edges].
            num_embeddings: Integer representing the number of embeddings.

        Returns:
            Positional embeddings as a torch tensor
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(
                0, num_embeddings, 2, dtype=torch.float32, device=self.device
            )
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        """Compute orientations between pairs of atoms from neighboring residues.

        Args:
            X: torch.Tensor representing atom coordinates with shape [n_atoms, 3]

        Returns:
            torch.Tensor atom pair orientations
        """
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        """Compute the unit vector representing the imputed side chain directions (C_beta - C_alpha).

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]

        Returns:
            Torch tensor representing side chain directions with shape [seq_len, 3]
        """
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

class StandardProteinGraphDataset(BaseProteinGraphDataset):
    """
    Take care of encoding non-standard AA (represented as "X") in
    self.letter_to_num
    """

    def __init__(self, data_list, **kwargs):
        """
        Initializes the dataset

        Args:
            data_list: List containint the initial data

        Returns:
            None
        """
        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
            "X": 0,
        }
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        super(StandardProteinGraphDataset, self).__init__(data_list, **kwargs)

    def _featurize_as_graph(self, protein):
        """Featurizes the protein information as a graph for the GNN

        Args:
            protein: Dictionary with the protein seq, coord and name.

        Returns:
            Torch geometric data instance representing with the protein information
        """
        name = protein["name"]
        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )
            seq = torch.as_tensor(
                [self.letter_to_num[a] for a in protein["seq"]],
                device=self.device,
                dtype=torch.long,
            )

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        return torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
        )
    
class ProteinGraphDataset(BaseProteinGraphDataset):
    """
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary
    -style protein structures into featurized protein graphs as described in
    the GVP manuscript (https://arxiv.org/abs/2009.01411). Modified for
    ProtBERT feature extractor.

    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are
        excluded from message passing
    -input_ids  amino acid id encoding from ProtBERT tokenizer
    -attention_mask attention_mask from ProtBERT tokenizer

    Modified from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py.
    """

    def __init__(self, data_list, **kwargs):
        """
        Initializes the dataset

        Args:
            data_list: List containint the initial data

        Returns:
            None
        """
        super(ProteinGraphDataset, self).__init__(data_list, **kwargs)

    def _featurize_as_graph(self, protein):
        """Featurizes the protein information as a graph for the GNN

        Args:
            protein: Dictionary with the protein seq, coord and name.

        Returns:
            Torch geometric data instance representing with the protein information
        """
        name = protein["name"]
        #input_ids = protein["seq"]
        #attention_mask = protein["mask"]
        input_ids = None
        attention_mask = None

        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        data = torch_geometric.data.Data(
            x=X_ca,
            input_ids=input_ids,
            attention_mask=attention_mask,
            name=name,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
        )
        return data
    
class ProteinGraphDatasetWithTarget(StandardProteinGraphDataset):
    """Thin wrapper for ProteinGraphDataset to include targets.
    Intended for all (structure-only) GNN models."""

    def __init__(self, data_list, **kwargs):
        super(ProteinGraphDatasetWithTarget, self).__init__(
            data_list, **kwargs
        )

    def __getitem__(self, i):
        if not isinstance(self.data_list[i], tuple):
            self.data_list[i] = (
                self._featurize_as_graph(self.data_list[i]),
                self.data_list[i]["target"],
            )
        return self.data_list[i]

    def _preprocess(self):
        """Preprocess all the records in `data_list` with
        `_featurize_as_graph`"""
        for i in tqdm.tqdm(range(len(self.data_list))):
            self.data_list[i] = (
                self._featurize_as_graph(self.data_list[i]),
                self.data_list[i]["target"],
            )

class BertProteinGraphDatasetWithTarget(ProteinGraphDataset):
    """Thin wrapper for ProteinGraphDataset to include targets.
    Intended for all BERT+GNN models"""

    def __init__(self, data_list, **kwargs):
        """Initializes the dataset

        Args:
            data_list: a list of data records (dicts with `input_ids`, `attention_mask`, `target`)

        Return:
            None
        """
        super(BertProteinGraphDatasetWithTarget, self).__init__(
            data_list, **kwargs
        )

    def __getitem__(self, idx):
        """Retrieve protein information by index.

        Args:
            idx: Integer representing the position of the protein.

        Return:
            None
        """
        if not isinstance(self.data_list[idx], tuple):
            self.data_list[idx] = (
                self._featurize_as_graph(self.data_list[idx]),
                self.data_list[idx]["target"],
            )
        return self.data_list[idx]

    def _preprocess(self):
        """Preprocess all the records in `data_list` with
        `_featurize_as_graph`. Directly modifies self.data_list

        Args:

        Returns:
            None
        """
        for i in tqdm.tqdm(range(len(self.data_list))):
            self.data_list[i] = (
                self._featurize_as_graph(self.data_list[i]),
                self.data_list[i]["target"],
            )