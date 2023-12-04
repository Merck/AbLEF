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

""" Antibody Language Ensemble Fusion (AbLEF), a neural network for recursive fusion of antibody structural ensembles & language. """

import torch
import torch.nn.functional as F
import sys
from einops import rearrange, repeat
from torch import nn
import pdb
import ablang
from transformers import BertModel
from gvp.models import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv
import os
from typing import Tuple

def _ablang_forward(
    ablang_model=None,
    tokens=None,
    attention_masks=None
):
    """forward pass throught pretrained ablang to get token-level embeddings
    Arguments:
        ablang_model: pretrained antibody language model 'heavy' or  'light' chain
        embeding_dim: dimension of the embeddings (ablang = 768)
        tokens: tokenizations of the input sequences
    Returns:
        output token-level residue embeddings (B, SEQ_L = 160, 768)
    Sources:
        language model backpropogation: https://github.com/aws-samples/lm-gvp
        ablang model: https://github.com/oxpig/AbLang
    """
    res_embeddings = ablang_model.AbRep(tokens, attention_masks).last_hidden_states[:, 1:-1, :]
    
    return res_embeddings

def _freeze_ablang(
    ablang_model=None, freeze_ablang=True, freeze_layer_count=-1
):
    """freeze pretrained parameters in AbLang
    Arguments:
        ablang_model: heavy or light chain
        freeze: Bool to freeze the pretrained ablang model
        freeze_layer_count: If freeze == False, number of layers to freeze (max_layers = 12).
    Returns:
        ablang_hc or ablang_lc model
    Sources:
        language model backpropogation: https://github.com/aregre-samples/lm-gvp
        ablang model: https://github.com/oxpig/AbLang
    """
    if freeze_ablang:
        # freeze the entire ablang model
        for param in ablang_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in ablang_model.AbRep.AbEmbeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in ablang_model.AbRep.EncoderBlocks.Layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None

def _bert_forward(
    bert_model: BertModel,
    embedding_dim: int,
    tokens=None,
    attention_masks=None
):
    """forward pass through pretrained BERT (protbert or protbert-bfd) to get token-level embeddings
    Arguments:
        bert_model: pretrained general protein language model (protbert ~200m seqs & protbert-bfd ~2b seqs)
        embeding_dim: dimension of the embeddings (bert = 1024)
        tokens: tokenizations of the input sequences
    Returns:
        output token-level residue embeddings (B, SEQ_L = 160, 1024)
    Sources:
        language model backpropogation: https://github.com/aws-samples/lm-gvp
        protbert model: https://huggingface.co/Rostlab/prot_bert
        protbert-bfd model: https://huggingface.co/Rostlab/prot_bert_bfd
    """
    # skip [CLS] and [SEP]
    res_embeddings = bert_model(
        tokens, attention_mask=attention_masks
    ).last_hidden_state[:, 1:-1, :]

    return res_embeddings

def _freeze_bert(
    bert_model: BertModel, freeze_bert=True, freeze_layer_count=-1
):
    """freeze pretrained parameters in BertModel
    Args:
        bert_model: HuggingFace bert model
        freeze: Bool to freeze the bert models
        freeze_layer_count: If freeze_bert == False, freeze up to this layer (max_layers = 30).
    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # 1/sqrt(64)=0.125
        self.to_qkv = nn.Linear(dim, dim*3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, maps = None, fusion_size=None):
        b, n, _, h = *x.shape, self.heads # b:batch_size, n:17, _:64, heads:heads as an example
        qkv = self.to_qkv(x).chunk(3, dim = -1) # self.to_qkv(x) to generate [b=batch_size, n=17, hd=192]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # q, k, v [b=batch_size, heads=heads, n=17, d=depth]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # [b=batch_size, heads=heads, 17, 17]

        mask_value = -torch.finfo(dots.dtype).max # A big negative number

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True) # [b=batch_size, 17]
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions' # mask [4, 17], dots [4, 8, 17, 17]
            assert len(mask.shape) == 2
            dots = dots.view(-1, fusion_size*fusion_size, dots.shape[1], dots.shape[2], dots.shape[3])
            mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            dots = dots * mask + mask_value * (1 - mask)
            dots = dots.view(-1, dots.shape[2], dots.shape[3], dots.shape[4])
            del mask

        if maps is not None:
            maps = F.pad(maps.flatten(1), (1, 0), value = 1.)
            maps = maps.unsqueeze(1).unsqueeze(2)
            dots.masked_fill_(~maps.bool(), mask_value)
            del maps

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        #print(out.shape)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None, maps = None, fusion_size=None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, maps = maps, fusion_size = fusion_size)
            x = ff(x)
        return x

class SuperResTransformer(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

    def forward(self, img, mask = None, maps= None, fusion_size = None):
        b, n, _ = img.shape
        # No need to add position code, just add token
        features_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((features_token, img), dim=1)
        x = self.transformer(x, mask, maps, fusion_size)
        x = self.to_cls_token(x[:, 0])

        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_hidden_fusion=64, kernel_size=3):
        '''
        Args:
            n_hidden_fusion : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        residual = self.block(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''
        super(Encoder, self).__init__()

        in_channels = setup["in_channels"]
        num_layers = setup["num_layers"]
        kernel_size = setup["kernel_size"]
        n_hidden_fusion = setup["n_hidden_fusion"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(n_hidden_fusion, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x

class Decoder(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''
        super(Decoder, self).__init__()

        self.final = nn.Sequential(nn.Conv2d(in_channels=setup["encoder"]["n_hidden_fusion"],
                                             out_channels=1,
                                             kernel_size=setup["decoder"]["kernel_size"],
                                             padding=setup["decoder"]["kernel_size"] // 2),
                     nn.PReLU())

        #self.pixelshuffle = nn.PixelShuffle(1)

    def forward(self, x):
 
        x = self.final(x)
        #print('decoder')
        #print(x.shape)
        #x = self.pixelshuffle(x)
       # print(x.shape)

        return x

class LangGVP(nn.Module):
    """Language + GVP-GNN structure fusion (modified from LM-GVP).

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    """

    def __init__(self, setup):
        """
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            weights: a tensor of class weights
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            freeze: wheter to freeze the entire langauge model
            freeze_layer_count: number of language embedding layers to freeze

        Returns:
            None
        """

        super(LangGVP, self).__init__()

        self.node_in_dim = (6,3)
        self.node_h_dim = (256,16)
        self.edge_in_dim = (32,1)
        self.edge_h_dim = (32,1)
        self.num_layers = 3
        self.drop_rate = setup['drop_rate']
        self.num_predictions = setup['num_predictions']
        self.first_chain = setup['first_chain']
        self.cdr_patch = setup['cdr_patch']
        self.seq_len = 160
        self.model = setup['model']
        self.freeze = setup['freeze']
        self.freeze_layer_count = setup['freeze_layer_count']
        self.identity = nn.Identity()
        self.residual = False

        if setup['model'] == 'ablang':
            self.ablang_lc = ablang.pretrained("light").AbLang.train()
            self.ablang_hc = ablang.pretrained("heavy").AbLang.train()
            self.embedding_dim = 768
            _freeze_ablang(
            self.ablang_lc,
            freeze_ablang = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count']
            )
            _freeze_ablang(
                self.ablang_hc,
                freeze_ablang = setup['freeze'],
                freeze_layer_count = setup['freeze_layer_count']
            )

        elif setup['model'] == 'protbert':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features

            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
        
    
        elif setup['model'] == 'protbert-bfd':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features
            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
        
        elif setup['model'] == 'ablang-only' or setup['model'] == 'protbert-only' or setup['model'] == 'protbert-bfd-only' or setup['model'] == 'none':
            raise ValueError('Set graph == False for ablang-only, protbert-only, protbert-bfd-only, or none models')

        node_in_dim = (self.node_in_dim[0] + self.embedding_dim, self.node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, self.node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=self.drop_rate)
            for _ in range(self.num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                self.node_h_dim[0] * self.num_layers,
                self.node_h_dim[1] * self.num_layers,
            )
        else: 
            node_h_dim = self.node_h_dim

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0))
        )

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(2 * ns, self.num_predictions),
        )


    def forward(self, lrs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=None):
        """
        Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
            input_ids: IDs of the embeddings to be used in the model.
        Returns:
            logits
        """
        
        h_V = (lrs.node_s, lrs.node_v)
        h_E = (lrs.edge_s, lrs.edge_v)

        edge_index = lrs.edge_index

        n_nodes = lrs.num_nodes
        batch_size = lrs.num_graphs
        #lc_tokens = lc_tokens
        lc_attention_masks = lc_attention_masks
        hc_tokens = hc_tokens
        hc_attention_masks = hc_attention_masks
        hc_tokens = hc_tokens
        hc_attention_masks = hc_attention_masks

        if self.model == 'ablang':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 768)
            lc_embeddings = _ablang_forward(
                self.ablang_lc,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 768)
            hc_embeddings = _ablang_forward(
                self.ablang_hc,
                hc_tokens, hc_attention_masks)
        elif self.model == 'protbert' or self.model == 'protbert-bfd':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            lc_embeddings = _bert_forward(
                self.bert_lc, self.embedding_dim,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            hc_embeddings = _bert_forward(
                self.bert_hc, self.embedding_dim,
                hc_tokens, hc_attention_masks)
            
        if self.first_chain == "H":
            attention_mask = torch.cat([hc_attention_masks[:,:-1],lc_attention_masks[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768/1024 (ablang/protbert))
            fv_embeddings = fv_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 0]
        elif self.first_chain == "L":
            attention_mask = torch.cat([lc_attention_masks[:,:-1], hc_attention_masks[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768/1024 (ablang/protbert))
            fv_embeddings = fv_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 0]
        else:
            raise ValueError("first_chain must be either 'H' or 'L'")
        
        node_embeddings = self.identity(fv_embeddings)

        h_V = (torch.cat([h_V[0], node_embeddings], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)

        out = scatter_mean(out, lrs.batch, dim=0)
        p = self.dense(out).squeeze(-1) + 0.5
        return p.unsqueeze(1)  # [bs, 1]

class LangGAT(nn.Module):
    def __init__(self, setup):
        """Language + GAT structure fusion. (modified from LM-GVP)


        Args:
            setup : dict, setup file (i.e., setup/setup.json)

        Returns:
            None
        """
        super(LangGAT, self).__init__()

        self.drop_rate = setup['drop_rate']
        self.num_predictions = setup['num_predictions']
        self.first_chain = setup['first_chain']
        self.cdr_patch = setup['cdr_patch']
        self.seq_len = 160
        self.n_hidden = 512
        self.model = setup['model']
        self.freeze = setup['freeze']
        self.freeze_layer_count = setup['freeze_layer_count']

        if setup['model'] == 'ablang':
            self.ablang_lc = ablang.pretrained("light").AbLang.train()
            self.ablang_hc = ablang.pretrained("heavy").AbLang.train()
            self.embedding_dim = 768
            _freeze_ablang(
            self.ablang_lc,
            freeze_ablang = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count']
            )
            _freeze_ablang(
                self.ablang_hc,
                freeze_ablang = setup['freeze'],
                freeze_layer_count = setup['freeze_layer_count']
            )

        elif setup['model'] == 'protbert':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features

            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
        
    
        elif setup['model'] == 'protbert-bfd':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features
            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
        
        
        elif setup['model'] == 'ablang-only' or setup['model'] == 'protbert-only' or setup['model'] == 'protbert-bfd-only' or setup['model'] == 'none':
            raise ValueError('Set graph == False for ablang-only, protbert-only, protbert-bfd-only, or none models')

        self.conv1 = GATConv(self.embedding_dim, 128, 4)
        self.conv2 = GATConv(512, 128, 4)
        self.conv3 = GATConv(512, 256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

        self.dense = nn.Sequential(
            nn.Linear(2048, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_predictions),
        )

    def forward(self, lrs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=None): 

        edge_index = lrs.edge_index
        n_nodes = lrs.num_nodes
        batch_size = lrs.num_graphs

        if self.model == 'ablang':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 768)
            lc_embeddings = _ablang_forward(
                self.ablang_lc,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 768)
            hc_embeddings = _ablang_forward(
                self.ablang_hc,
                hc_tokens, hc_attention_masks)
        elif self.model == 'protbert' or self.model == 'protbert-bfd':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            lc_embeddings = _bert_forward(
                self.bert_lc, self.embedding_dim,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            hc_embeddings = _bert_forward(
                self.bert_hc, self.embedding_dim,
                hc_tokens, hc_attention_masks)
            
        if self.first_chain == "H":
            attention_mask = torch.cat([hc_attention_masks[:,:-1],lc_attention_masks[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768/1024 (ablang/protbert))
            fv_embeddings = fv_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 0]
        elif self.first_chain == "L":
            attention_mask = torch.cat([lc_attention_masks[:,:-1], hc_attention_masks[:,1:]],1)
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768/1024 (ablang/protbert))
            fv_embeddings = fv_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 0]
        else:
            raise ValueError("first_chain must be either 'H' or 'L'")
        
         # GAT forward
        conv1_out = self.conv1(fv_embeddings, edge_index)
        conv2_out = self.conv2(conv1_out, edge_index)
        conv3_out = self.conv3(conv2_out, edge_index)
        # residual concat
        out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]
        # aggregate node vectors to graph
        out = scatter_mean(out, lrs.batch, dim=0)  # [bs, 2048]
        p = self.dense(out).squeeze(-1) + 0.5
        return p.unsqueeze(1)  # [bs, 1]

    
class LangEnsemble(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setup file (i.e., setup/setup.json)
        '''
        super(LangEnsemble, self).__init__()
        self.drop_rate = setup['drop_rate']
        self.num_predictions = setup['num_predictions']
        self.first_chain = setup['first_chain']
        self.model = setup['model']
        self.cdr_patch = setup['cdr_patch']
        self.seq_len = 160

        if self.cdr_patch == False:
            self.fusion_size = 320
        elif self.cdr_patch == "cdrs":
            self.fusion_size = 128
            self.lng_size = 64
        elif self.cdr_patch == "cdr3s":
            self.fusion_size = 64
            self.lng_size = 32
        
        if setup['model'] == 'ablang' or setup['model'] == 'ablang-only':
            self.ablang_lc = ablang.pretrained("light").AbLang.train()
            self.ablang_hc = ablang.pretrained("heavy").AbLang.train()
            self.embedding_dim = 768
            _freeze_ablang(
            self.ablang_lc,
            freeze_ablang = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count']
            )
            _freeze_ablang(
                self.ablang_hc,
                freeze_ablang = setup['freeze'],
                freeze_layer_count = setup['freeze_layer_count']
            )

        elif setup['model'] == 'protbert' or setup['model'] == 'protbert-only':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features

            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
        
    
        elif setup['model'] == 'protbert-bfd' or setup['model'] == 'protbert-bfd-only':
            self.bert_lc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.bert_hc = BertModel.from_pretrained('config/bert/prot_bert_bfd')
            self.embedding_dim = self.bert_lc.pooler.dense.out_features
            _freeze_bert(
                self.bert_lc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
            _freeze_bert(
                self.bert_hc, 
                freeze_bert=setup['freeze'], 
                freeze_layer_count=setup['freeze_layer_count'])
    
            
        elif setup['model'] == 'none':
            self.embedding_dim = 1024


        if setup['model'] == 'ablang-only' or setup['model'] == 'protbert-only' or setup['model'] == 'protbert-bfd-only':
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=self.drop_rate) 
            self.dense = nn.Sequential(
                nn.Linear(self.fusion_size * self.embedding_dim, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            )
        elif setup['model'] == 'ablang' or setup['model'] == 'protbert' or setup['model'] == 'protbert-bfd' :
            if self.fusion_size == 320:
                self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p=self.drop_rate)
                self.dense = nn.Sequential(
                nn.Linear(2 * self.embedding_dim * self.fusion_size, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            ) 
            else:
                self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
                self.lngdense = nn.Linear(self.seq_len, self.lng_size)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p=self.drop_rate)
                self.dense = nn.Sequential(
                nn.Linear(2 * self.embedding_dim * self.fusion_size, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            )
        elif setup['model'] == 'none':
            if self.fusion_size == 320:
                self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p=self.drop_rate)
                self.dense = nn.Sequential(
                nn.Linear(self.fusion_size * self.embedding_dim, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            )
            else:
                self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p=self.drop_rate)
                self.dense = nn.Sequential(
                nn.Linear(self.fusion_size * self.embedding_dim, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            )
        
    def forward(self, srs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=None):                                  
        '''
        Combines ensemble fusion with ablang embeddings as an input                                                                                                   tensor x.
        Args:
            x : tensor (B, SRS, LC_SEQ, HC_SEQ), input ensemble fusion & seqs
        Returns:
            out: tensor (B, PROP), hidden states
        
        '''
        # ensemble fusion --> hidden layer ablang
        if srs is not None:
            ensemble_fusions = srs
            batch_size, heigth, width = ensemble_fusions.shape
        else:
            batch_size, length = lc_tokens.shape

        

        # push sequences
        lc_tokens = lc_tokens
        lc_attention_masks = lc_attention_masks
        hc_tokens = hc_tokens
        hc_attention_masks = hc_attention_masks

        if self.model == 'ablang' or self.model == 'ablang-only':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 768)
            lc_embeddings = _ablang_forward(
                self.ablang_lc,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 768)
            hc_embeddings = _ablang_forward(
                self.ablang_hc,
                hc_tokens, hc_attention_masks)
            
            if self.model == 'ablang':
                srs_embeddings = self.ensdense(ensemble_fusions)
                if self.fusion_size == 320:
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
                else:
                    hc_embeddings = hc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    lc_embeddings = lc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    hc_embeddings = self.lngdense(hc_embeddings)
                    lc_embeddings = self.lngdense(lc_embeddings)
                    hc_embeddings = hc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    lc_embeddings = lc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
                    
            elif self.model == 'ablang-only':
                if self.fusion_size == 320:
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
                else:
                    hc_embeddings = hc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    lc_embeddings = lc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    hc_embeddings = self.lngdense(hc_embeddings)
                    lc_embeddings = self.lngdense(lc_embeddings)
                    hc_embeddings = hc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    lc_embeddings = lc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdr) or 64 (cdr3), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdr) or 64 (cdr3), LANG_EMBEDDING = 768 (ablang))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")


        elif self.model == 'protbert' or self.model == 'protbert-bfd' or self.model == 'protbert-only' or self.model == 'protbert-bfd-only':
            # (B, LC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            lc_embeddings = _bert_forward(
                self.bert_lc, self.embedding_dim,
                lc_tokens, lc_attention_masks)
            
            # (B, HC_SEQ = 160, ProtBert_EMBEDDING = 1024)
            hc_embeddings = _bert_forward(
                self.bert_hc, self.embedding_dim,
                hc_tokens, hc_attention_masks)
            
            if self.model == 'protbert' or self.model == 'protbert-bfd':
                srs_embeddings = self.ensdense(ensemble_fusions)
                if self.fusion_size == 320:
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
                else:
                    hc_embeddings = hc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    lc_embeddings = lc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    hc_embeddings = self.lngdense(hc_embeddings)
                    lc_embeddings = self.lngdense(lc_embeddings)
                    hc_embeddings = hc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    lc_embeddings = lc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = torch.cat((srs_embeddings, fv_embeddings), dim =2) # (B, LCSEQ + HCSEQ = 128 (cdrs) or 64 (cdr3s), LANG_EMBEDDING + SRS_EMBEDDING)
                        ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
            elif self.model == 'protbert-only' or self.model == 'protbert-bfd-only':
                if self.fusion_size == 320:
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 320 (false), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
                else:
                    hc_embeddings = hc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    lc_embeddings = lc_embeddings.view(batch_size, self.embedding_dim, self.seq_len)
                    hc_embeddings = self.lngdense(hc_embeddings)
                    lc_embeddings = self.lngdense(lc_embeddings)
                    hc_embeddings = hc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    lc_embeddings = lc_embeddings.view(batch_size, self.lng_size, self.embedding_dim)
                    if self.first_chain == "H":
                        fv_embeddings = torch.cat((hc_embeddings, lc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdr) or 64 (cdr3), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    elif self.first_chain == "L":
                        fv_embeddings = torch.cat((lc_embeddings, hc_embeddings), dim=1) # (B, LCSEQ + HCSEQ = 128 (cdr) or 64 (cdr3), LANG_EMBEDDING = 1024 (bert))
                        ens_lang = fv_embeddings.view(batch_size, -1)
                    else:
                        raise ValueError("first_chain must be either 'H' or 'L'")
        elif self.model == 'none':
            srs_embeddings = self.ensdense(ensemble_fusions)
            ens_lang = srs_embeddings.view(batch_size, -1) # (B, [SRS_EMBEDDING] = 245760)

        p = self.relu (ens_lang)
        p = self.dropout (p)
        p = self.dense(p)

        return p

class ALEFNet(nn.Module):
    ''' Antibody Language Ensemble Fusion Net, a neural network for recursive fusion of antibody structural ensembles & language. '''

    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''

        super(ALEFNet, self).__init__()
        if setup["language"]["graph"] == False:
            self.encode = Encoder(setup["encoder"])
            self.superres = SuperResTransformer(dim=setup["encoder"]["n_hidden_fusion"],
                                                depth=setup["transformer"]["depth"],
                                                heads=setup["transformer"]["heads"],
                                                mlp_dim=setup["transformer"]["mlp_dim"],
                                                dropout=setup["transformer"]["dropout"])
            self.decode = Decoder(setup)
            self.property = LangEnsemble(setup["language"])
            self.GVPproperty = None
            self.GATproperty = None
        elif setup["language"]["graph"] == "GVP":
            self.GVPproperty = LangGVP(setup["language"])
            self.GATproperty = None
        elif setup["language"]["graph"] == "GAT":
            self.GATproperty = LangGAT(setup["language"])
            self.GVPproperty = None
        else:
            raise ValueError("graph must be either 'GVP' or 'GAT' or false in setup.json")


    def forward(self, lrs, ens_attention_masks, alphas, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size):
        '''
        Super resolves a batch of low-resolution ensembles, integrates pretrained language model, and predicts antibody property.
        Args:
            lrs : tensor (B, C, L, W, H), low-resolution ensemble
            ens_attention_masks: tensor (B, L, W, H), ensemble attention masks from padded sequences
            alphas : tensor (B, L), boolean indicator (0 if collated ensemble, 1 otherwise)
            lc_tokens: tensor (B, LCSEQ), tokenized light chain sequences (ProtBERT convention)
            lc_attention_masks: tensor (B, LCSEQ) tokenized light chain attention masks (ProtBERT convention)
            hc_tokens: tesnor (B, HCSEQ) tokenized heavy chain sequences (ProtBERT convention)
            hc_attention_masks: tensor (B, HCSEQ) tokenized heavy chain attention masks (ProtBERT convention)
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved ensemble
            prop: tensor(B, PROP), predicted property of antibody
        '''

        if self.GVPproperty is not None:
            prop = self.GVPproperty(lrs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=fusion_size)
        elif self.GATproperty is not None:
            prop = self.GATproperty(lrs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=fusion_size)
        elif lrs is not None and self.GVPproperty is None and self.GATproperty is None:
            batch_size, channel_len, seq_len, heigth, width = lrs.shape
            stacked_input = lrs.view(batch_size * seq_len, channel_len, width, heigth)
            layer1 = self.encode(stacked_input) # encode input tensor

            ####################### encode ensemble ######################
            layer1 = layer1.view(batch_size, seq_len, -1, width, heigth) # tensor (B, L, C, W, H)
            ####################### fuse ensemble ######################
            img = layer1.permute(0, 3, 4, 1, 2).reshape(-1, layer1.shape[1], layer1.shape[2])  # .contiguous().view == .reshape()
            if ens_attention_masks is not None:
                ens_attention_masks = ens_attention_masks.permute(0, 2, 3, 1).reshape(-1, ens_attention_masks.shape[1])
            preds = self.superres(img, mask=alphas, maps=ens_attention_masks, fusion_size=fusion_size)
            preds = preds.view(-1, width, heigth, preds.shape[-1]).permute(0, 3, 1, 2)
            ####################### decode ensemble ######################
            srs = self.decode(preds)  # decode final hidden state (B, C_out, W, H)
            srs = srs.squeeze(1)
            ################# predict property (ensemble + language embeddings) ############
            prop = self.property(srs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=fusion_size)
        else: 
            srs = None
            prop = self.property(srs, lc_tokens, lc_attention_masks, hc_tokens, hc_attention_masks, fusion_size=fusion_size)
        return prop
