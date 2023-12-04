#%%
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
import json
import os
import argparse
from collections import defaultdict
import pandas as pd
import torch

# this script save the graph structure of each protein in train and holdout data directories as .pt files
os.chdir('pathway/to/graph_structs.py')
df = pd.read_csv('fv_data.csv')
# slice train data from train_dataset
train_dataset = df[df['split'] == 'train']
holdout_dataset = df[df['split'] == 'holdout']

# read json proteins_train and proteins_holdout
with open('proteins_train.json', 'r') as f:
    proteins_train = json.load(f)
with open('proteins_holdout.json', 'r') as f:
    proteins_holdout = json.load(f)

train = get_dataset(proteins_train)
holdout = get_dataset(proteins_holdout)

from pdb_pyg import get_dataset

cwd = os.getcwd()

for i in range(len(train)):
    print(train[i][0])
    #print(train_dataset['structure_path'].iloc[i])
    dirs_name = train_dataset['structure_path'].iloc[i].split('roidDIST/')[1]
    save_dir = train_dataset['structure_path'].iloc[i].split('roidDIST/')[0]
    dirs_name = dirs_name.split('/')[0]
    save_dir = save_dir + 'roidDIST/' + dirs_name
    save_name = dirs_name + '_01' + '.pt'
    os.chdir(save_dir)
    torch.save(train[i][0], save_name)
    os.chdir(cwd)
    
# same for holdout

for i in range(len(holdout)):
    print(holdout[i][0])
    #print(holdout_dataset['structure_path'].iloc[i])
    dirs_name = holdout_dataset['structure_path'].iloc[i].split('roidDIST/')[1]
    save_dir = holdout_dataset['structure_path'].iloc[i].split('roidDIST/')[0]
    dirs_name = dirs_name.split('/')[0]
    save_dir = save_dir + 'roidDIST/' + dirs_name
    save_name = dirs_name + '_01' + '.pt'
    os.chdir(save_dir)
    torch.save(holdout[i][0], save_name)
    os.chdir(cwd)

