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

""" script to load, augment, and preprocess batches of data """
from collections import OrderedDict
import numpy as np
import pandas as pd
import math
import torch
import os
from os.path import join, exists, dirname, basename, isfile
import sys
import heapq
import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb
import ablang
import torch_geometric
from transformers import BertTokenizer
import json

def get_patch(img, x, y, size=32):
    """
    Slices out a square patch from `ensemble structure` starting from the (x,y) top-left corner.
    If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
    and the output has shape (l, size, size).
    Args:
        img: numpy.ndarray (n, m), input image
        x, y: int, top-left corner of the patch
        size: int, patch size
    Returns:
        patch: numpy.ndarray (size, size)
    """
    patch = img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays
    return patch

class DataSet(OrderedDict):
    """
    An OrderedDict derived class to group the assets of a dataset.
    Sources: https://github.com/Suanmd/TR-MISR 
    """
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"
        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        #print(dict_info)
        return dict_info

def read_dataset(superset_dir, seed=None, ens_L=None, lang=None, graph=False, cdr_patch=False, resprop=False, median_ref=False):
    # read distance maps from channels
    lr_ens_c = []
    print(superset_dir)
    for dir_ in range(len(superset_dir)):
        if not graph:
            if cdr_patch == 'cdrs' and not lang == "ablang-only" and not lang == "protbert-only" and not lang == "protbert-bfd-only":
                idx_names = np.array([basename(path)[8:-4] for path in glob.glob(join(superset_dir[dir_], 'cdrs_DAB*.npy'))])
                idx_names = np.sort(idx_names)
                lr_ens = np.array([np.load(join(superset_dir[dir_], f'cdrs_DAB{i}.npy')) for i in idx_names], dtype=np.float32)
                ens_len, ens_h, ens_w  = lr_ens.shape
            elif cdr_patch == 'cdr3s' and not lang == "ablang-only" and not lang == "protbert-only" and not lang == "protbert-bfd-only":
                idx_names = np.array([basename(path)[9:-4] for path in glob.glob(join(superset_dir[dir_], 'cdr3s_DAB*.npy'))])
                idx_names = np.sort(idx_names)
                lr_ens = np.array([np.load(join(superset_dir[dir_], f'cdr3s_DAB{i}.npy')) for i in idx_names], dtype=np.float32)
                ens_len, ens_h, ens_w  = lr_ens.shape
            elif cdr_patch == False and not lang == "ablang-only" and not lang == "protbert-only" and not lang == "protbert-bfd-only":
                idx_names = np.array([basename(path)[3:-4] for path in glob.glob(join(superset_dir[dir_], 'DAB*.npy'))])
                idx_names = np.sort(idx_names)
                lr_ens = np.array([np.load(join(superset_dir[dir_], f'DAB{i}.npy')) for i in idx_names], dtype=np.float32)
                ens_len, ens_h, ens_w  = lr_ens.shape
            elif lang == "ablang-only" or lang == "protbert-only" or lang == "protbert-bfd-only":
                lr_ens = None
            else:
                raise ValueError('Invalid cdr_patch argument (must be false, "cdrs", or "cdr3s")')
        else:
            idx_names = np.array([basename(path)[3:-3] for path in glob.glob(join(superset_dir[dir_], 'DAB*.pt'))])
            idx_names = np.sort(idx_names)
            lr_ens = [torch.load(join(superset_dir[dir_], f'DAB{i}.pt')) for i in idx_names]

        

        if lang == "ablang" or lang == "ablang-only":
            # collect HC & LC sequences --> tokenize (ProtBERT convention)
            tokenizer = ablang.ABtokenizer(os.path.join("config/","vocab.json"))
            lc_seq = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'L_DAB*.txt'))[0], dtype='str').tolist()]
            hc_seq = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'H_DAB*.txt'))[0], dtype='str').tolist()]
            ## tokenize
            lc_token = tokenizer(lc_seq, pad=True)
            #lc_attention_mask = torch.ones(lc_token.shape, device = lc_token.device).masked_fill(lc_token == 21, 0).masked_fill(lc_token == 0, 0).masked_fill(lc_token == 22, 0)
            lc_attention_mask = torch.zeros(lc_token.shape, device = lc_token.device).masked_fill(lc_token == 21, 1)
            hc_token = tokenizer(hc_seq, pad=True)
            #hc_attention_mask = torch.ones(hc_token.shape, device = hc_token.device).masked_fill(hc_token == 21, 0).masked_fill(lc_token == 0, 0).masked_fill(lc_token == 22, 0)
            hc_attention_mask = torch.zeros(hc_token.shape, device = hc_token.device).masked_fill(hc_token == 21, 1)


        elif lang == "protbert" or lang == "protbert-only" or lang == "protbert-bfd" or lang == "protbert-bfd-only" :
            #import requests
            #requests.get('http://huggingface.co', verify=False)
            os.environ['CURL_CA_BUNDLE'] = ''
            if lang == "protbert" or lang == "protbert-only":
                tokenizer = BertTokenizer.from_pretrained("config/bert/prot_bert", do_lower_case=False)
                lc_seq2 = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'L_DAB*.txt'))[0], dtype='str').tolist()]
                hc_seq2 = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'H_DAB*.txt'))[0], dtype='str').tolist()]
                # replace '-' token with '[PAD]' token in seqs
                lc_seq2 = [[x if x != '-' else '[PAD]' for x in lc_seq2[0]]]
                hc_seq2 = [[x if x != '-' else '[PAD]' for x in hc_seq2[0]]]
                # combine seqs into string separated by space
                lc_seq = " ".join(lc_seq2[0])
                hc_seq = " ".join(hc_seq2[0])
                lc_token2 = tokenizer(lc_seq, padding=True, return_tensors="pt")
                lc_token, lc_attention_mask = lc_token2['input_ids'], lc_token2['attention_mask']
                lc_attention_mask = torch.ones(lc_token.shape, device = lc_token.device).masked_fill(lc_token == 0, 0)
                hc_token2 = tokenizer(hc_seq, padding=True, return_tensors="pt")
                hc_token, hc_attention_mask = hc_token2['input_ids'], hc_token2['attention_mask']
                hc_attention_mask = torch.ones(hc_token.shape, device = hc_token.device).masked_fill(hc_token == 0, 0)
                
            elif lang == "protbert-bfd" or lang == "protbert-bfd-only":
                tokenizer = BertTokenizer.from_pretrained("config/bert/prot_bert_bfd", do_lower_case=False)
                lc_seq2 = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'L_DAB*.txt'))[0], dtype='str').tolist()]
                hc_seq2 = [np.genfromtxt(glob.glob(join(superset_dir[dir_], f'H_DAB*.txt'))[0], dtype='str').tolist()]
                lc_seq = " ".join(lc_seq2[0])
                hc_seq = " ".join(hc_seq2[0])
                lc_token2 = tokenizer(lc_seq, padding=True, return_tensors="pt")
                lc_token, lc_attention_mask = lc_token2['input_ids'], lc_token2['attention_mask']
                lc_attention_mask = torch.ones(lc_token.shape, device = lc_token.device).masked_fill(lc_token == 0, 0)
                hc_token2 = tokenizer(hc_seq, padding=True, return_tensors="pt")
                hc_token, hc_attention_mask = hc_token2['input_ids'], hc_token2['attention_mask']
                hc_attention_mask = torch.ones(hc_token.shape, device = hc_token.device).masked_fill(hc_token == 0, 0)
        elif lang == "none":
            lc_token = None
            hc_token = None
            lc_attention_mask = None
            hc_attention_mask = None
        else:
            raise ValueError('Invalid language argument (must be "ablang", "protbert", "protbert-bfd", "none", or "ablang-only", "protbert-only", "protbert-bfd-only")')

        if lang == "ablang-only" or lang == "protbert-only" or lang == "protbert-bfd-only":
            ens_attention_mask = None
        elif graph is not False:
            ens_attention_mask = np.array(0)
        elif graph == False:
            # attention masks for the ensemble of distance maps
            ens_attention_mask = np.zeros(lr_ens.shape)
            ens_attention_mask[lr_ens != 0] = 1
            ens_attention_mask = np.array(ens_attention_mask)

        # add residue property maps
        if resprop and not graph:
            if resprop and exists(join(superset_dir[dir_], 'resprop.npy')) and cdr_patch == False  or resprop and exists(join(superset_dir[dir_], 'cdrs_resprop.npy')) and cdr_patch == 'cdrs' or resprop and exists(join(superset_dir[dir_], 'cdr3s_resprop.npy')) and cdr_patch == 'cdr3s':
                if cdr_patch == 'cdrs':
                    resprop_map = np.array(np.load(join(superset_dir[dir_], 'cdrs_resprop.npy')), dtype=np.float32)
                elif cdr_patch == 'cdr3s':
                    resprop_map = np.array(np.load(join(superset_dir[dir_], 'cdr3s_resprop.npy')), dtype=np.float32)
                elif cdr_patch == False:
                    resprop_map = np.array(np.load(join(superset_dir[dir_], 'resprop.npy')), dtype=np.float32)

                if resprop == 'add':
                    for i in range(len(lr_ens)):
                        lr_ens[i] = lr_ens[i] + resprop_map
                    lr_ens_c.append(lr_ens)
                    
                elif resprop == 'channel' and dir_ == len(superset_dir)-1:
                    resprop_ens = np.stack([resprop_map]*ens_len, axis=0)
                    lr_ens_c.append(lr_ens)
                    lr_ens_c.append(resprop_ens)
                elif resprop == 'channel' and dir_ != len(superset_dir)-1:
                    lr_ens_c.append(lr_ens)
                    continue
                else:
                    raise ValueError("invalid ensemble residue property map option")

            elif resprop and not exists(join(superset_dir[dir_], 'resprop.npy')) and cdr_patch == False  or resprop and not exists(join(superset_dir[dir_], 'cdrs_resprop.npy')) and cdr_patch == 'cdrs' or resprop and not exists(join(superset_dir[dir_], 'cdr3s_resprop.npy')) and cdr_patch == 'cdr3s':
                prfx = cdr_patch + "_" if cdr_patch else ''
                dir_name = basename(superset_dir[dir_])
                if cdr_patch == 'cdrs' or cdr_patch == 'cdr3s':
                    lc_pth = join(superset_dir[dir_], 'L_%s.json' %dir_name)
                    lc_anarci = json.load(open(lc_pth))
                    hc_pth = join(superset_dir[dir_], 'H_%s.json' %dir_name)
                    hc_anarci = json.load(open(hc_pth))
                    cdr1_PAD, cdr2_PAD, cdr3_PAD = 20, 12, 32
                    cdrl1, cdrl2, cdrl3 = lc_anarci['cdr1'], lc_anarci['cdr2'], lc_anarci['cdr3']
                    cdrh1, cdrh2, cdrh3 = hc_anarci['cdr1'], hc_anarci['cdr2'], hc_anarci['cdr3']
                    cdrl1 = [lc_anarci['seq'][x] for x in cdrl1]
                    cdrl2 = [lc_anarci['seq'][x] for x in cdrl2]
                    cdrl3 = [lc_anarci['seq'][x] for x in cdrl3]
                    cdrh1 = [hc_anarci['seq'][x] for x in cdrh1]
                    cdrh2 = [hc_anarci['seq'][x] for x in cdrh2]
                    cdrh3 = [hc_anarci['seq'][x] for x in cdrh3]
                    cdrl1 = cdrl1 + ['-']*(cdr1_PAD-len(cdrl1))
                    cdrl2 = cdrl2 + ['-']*(cdr2_PAD-len(cdrl2))
                    cdrl3 = cdrl3 + ['-']*(cdr3_PAD-len(cdrl3))
                    cdrh1 = cdrh1 + ['-']*(cdr1_PAD-len(cdrh1))
                    cdrh2 = cdrh2 + ['-']*(cdr2_PAD-len(cdrh2))
                    cdrh3 = cdrh3 + ['-']*(cdr3_PAD-len(cdrh3))

                    cdrs = cdrl1 + cdrl2 + cdrl3 + cdrh1 + cdrh2 + cdrh3
                    cdrs = ''.join(cdrs)
                    cdr3s = cdrl3 + cdrh3
                    cdr3s = ''.join(cdr3s)
                    if cdr_patch == 'cdrs':
                        Fv = cdrs
                    elif cdr_patch == 'cdr3s':
                        Fv = cdr3s
                else:
                    Fv = str(np.char.add(lc_seq[0], hc_seq[0]))


                resprop_grid = pd.read_json(os.path.join("config/","resprop.json"))
                resprop_map = np.empty((ens_w, ens_h), dtype=float)
                for residue1 in range(len(Fv)):
                    for residue2 in range (len(Fv)):
                        residue_pair = Fv[residue1]+Fv[residue2]
                        pair_prop = resprop_grid[Fv[residue1]][Fv[residue2]]
                        pair_prop_inv = resprop_grid[Fv[residue2]][Fv[residue1]]
                        if math.isnan(pair_prop) is False:
                            resprop_map[residue1][residue2] = pair_prop
                        elif math.isnan(pair_prop_inv) is False:
                            resprop_map[residue1][residue2] = pair_prop_inv
                        else:
                            raise ValueError("residue pair not found")

                
                np.save(join(superset_dir[dir_], '%sresprop.npy') % prfx, resprop_map)

                if resprop == 'add':
                    for i in range(len(lr_ens)):
                        lr_ens[i] = lr_ens[i] + resprop_map
                    lr_ens_c.append(lr_ens)
                    
                elif resprop == 'channel' and dir_ == len(superset_dir)-1:
                    resprop_ens = np.stack([resprop_map]*ens_len, axis=0)
                    lr_ens_c.append(lr_ens)
                    lr_ens_c.append(resprop_ens)
                elif resprop == 'channel' and dir_ != len(superset_dir)-1:
                    lr_ens_c.append(lr_ens)
                    continue
                else:
                    raise ValueError("invalid ensemble residue property map option")
            else:
                lr_ens_c.append(lr_ens)
        elif resprop and graph:
            raise ValueError("residue property maps only for aLEF not graph representations")
        else:
            lr_ens_c.append(lr_ens) 

        # property to predict
        if exists(join(superset_dir[dir_], 'prop.npy')):
            prop = np.array(np.load(join(superset_dir[dir_], 'prop.npy')), dtype=np.float32)
        else:
            prop = None

        #if create_patches:
            #if seed is not None:
                #np.random.seed(seed)

        # max_x = lr_ens[0].shape[0] - patch_size
            #max_y = lr_ens[0].shape[1] - patch_size
            #x = np.random.randint(low=0, high=max_x)
            #y = np.random.randint(low=0, high=max_y)
            #lr_ens = get_patch(lr_ens, x, y, patch_size)  # broadcast slicing coordinates across all structures
            #ens_attention_mask = get_patch(ens_attention_mask, x, y, patch_size)

        if lang == "ablang-only" or lang == "protbert-only" or lang == "protbert-bfd-only" and graph == False:
            lr_ens_stacked = None
        elif graph != False:
            lr_ens_stacked = lr_ens_c
        else:
            lr_ens_stacked = np.stack(lr_ens_c, axis=0)

    if median_ref and not graph:
        c_stack, l_stack, w_stack, h_stack = lr_ens_stacked.shape
        refs = np.median(lr_ens_stacked[:,:ens_len,:,:], axis=1, keepdims=True)
        refs_stacked = np.repeat(refs, ens_len, axis=1)
        lr_ens_stacked = np.concatenate((lr_ens_stacked, refs_stacked), axis=0)
        lr_ens_stacked = np.array(lr_ens_stacked)

    if lang == "ablang-only" and not graph or lang == "protbert-only" and not graph or lang == "protbert-bfd-only" and not graph:
        lr_ens_stacked = None
    elif graph:
        lr_ens_stacked = lr_ens_c
    else:
        lr_ens_stacked = np.array(lr_ens_stacked)
    
    # organize all assets into a DataSet (OrderedDict)
    dataset = DataSet(name=basename(superset_dir[0]),
                        lr=lr_ens_stacked, ens_attention_mask = ens_attention_mask,
                        lc_token=lc_token, lc_attention_mask=lc_attention_mask,
                        hc_token=hc_token, hc_attention_mask=hc_attention_mask,
                        prop=prop)
    return dataset

class SuperSet(Dataset):
    """ Derived SuperSet class for loading many datasets from a list of directories."""

    def __init__(self, superset_dir, setup, seed=None, ens_L=-1, lang=None,  graph = False, cdr_patch=False, resprop=False, median_ref=False, **kwargs):

        super().__init__()
        self.superset_dir = superset_dir
        self.name_to_dir = defaultdict(list)
        for s_dir in superset_dir:
            self.name_to_dir[basename(s_dir)].append(s_dir)

        self.superset_dir = [[v] for k, v in self.name_to_dir.items()]
        #self.name_to_dir = {basename(dirname(s_dir)) + '-' + basename(s_dir): s_dir for s_dir in superset_dir}
        self.seed = seed  # seed for random patches
        self.ens_L = ens_L
        self.lang = lang
        self.graph = graph
        self.cdr_patch = cdr_patch
        self.resprop = resprop
        self.median_ref = median_ref


    def __len__(self):
        return len(self.superset_dir)        

    def __getitem__(self, index):
        """ Returns a dict of all assets in the directory of the given index."""    
        superset_dir = self.superset_dir[index]

        superset_dir = superset_dir[0]

        superset = [read_dataset(superset_dir=superset_dir,
                               seed=self.seed,
                               ens_L=self.ens_L, lang = self.lang, graph = self.graph, cdr_patch = self.cdr_patch, resprop = self.resprop, median_ref = self.median_ref)]

        if len(superset) == 1:
            superset = superset[0]
        
        superset_list = superset if isinstance(superset, list) else [superset]

        for i, superset_ in enumerate(superset_list):
            if superset_['lr'] is None:
                superset_['lr'] = superset_['lr']
                superset_['ens_attention_mask'] = superset_['ens_attention_mask']
            elif type(superset_['lr'][0][0]) is torch_geometric.data.data.Data:
                superset_['lr'] = superset_['lr'][0][0]
                superset_['ens_attention_mask'] = torch.from_numpy(superset_['ens_attention_mask'].astype(np.bool_))
            elif type(superset_['lr'][0]) is np.ndarray:
                superset_['lr'] = torch.from_numpy(superset_['lr'].astype(np.float32))
                superset_['ens_attention_mask'] = torch.from_numpy(superset_['ens_attention_mask'].astype(np.bool_))
            superset_['lc_token'] = superset_['lc_token']
            superset_['lc_attention_mask'] = superset_['lc_attention_mask']
            superset_['hc_token'] = superset_['hc_token']
            superset_['hc_attention_mask'] = superset_['hc_attention_mask']

            if superset_['prop'] is not None:
                superset_['prop'] = torch.from_numpy(superset_['prop'].astype(np.float32))
            superset_list[i] = superset_

        
        if len(superset_list) == 1:
            superset = superset_list[0]

        return superset