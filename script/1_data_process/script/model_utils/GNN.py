import os, sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.data import DGLDataset

from .NC import load_pickle

class CPGDataset(DGLDataset):
    def __init__(self,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):
        super(CPGDataset, self).__init__(raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        
        self.fl = os.listdir(self.raw_dir)


    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        nodes, edges = load_pickle(os.path.join(self.raw_dir, self.fl[idx]))
        cpg, cpg_aug = gen_aug_cpgs(nodes, edges)
        return cpg, cpg_aug

    def __len__(self):
        # number of data examples
        return len(self.fl)

def pretrain_collate_fn(batch):
    graphs, graphs_aug = list(zip(*batch))
    graphs = dgl.batch(graphs)
    graphs_aug = dgl.batch(graphs_aug)
    return graphs, graphs_aug
    # feats = graph.nodes['emb'].data['feat']
    # ast = dgl.edge_type_subgraph(graph, etypes=('emb', 'ast', 'emb'))
    # cpg = dgl.edge_type_subgraph(graph, 
    #                              etypes=[('emb', 'cfg', 'emb'),
    #                                      ('emb', 'cdg', 'emb'),
    #                                      ('emb', 'ddg', 'emb'),
    #                                      ])
    # return (feats, ast, cpg)

def get_pretrain_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
    )

def gen_cpg(nodes, edges):
    cpg_data = {
        ('emb', e.lower(), 'emb'): (torch.as_tensor(edges[e][0]), torch.as_tensor(edges[e][1])) for e in edges if len(edges[e]) > 0
    }
    cpg = dgl.heterograph(cpg_data)
    cpg.nodes['emb'].data['feat'] = torch.as_tensor(np.stack(nodes, axis=0))
    return cpg

def gen_aug_cpg(nodes: torch.Tensor, edges: torch.Tensor, p_node=0.2, p_edge=0.2)->tuple(dgl.DGLGraph, dgl.DGLGraph):
    # node augmentation + edge augmentation
    # node cutoff
    nodes_aug = nodes
    edges_aug = edges
    num = len(nodes)
    if num * p_node > 1:
        nodes_aug[torch.multinomial(torch.arange(0, num), num_samples=math.ceil(num*p_node))] = 0.
    for k in edges:
        if len(edges[k]) and len(edges[k][0]) * p_edge > 1:
            idxs = random.sample(range(len(edges[k][0])), math.ceil(len(edges[k][0]) * p_edge))
            edges_aug[k] = [[edges_aug[k][0][i] for i in range(len(edges_aug[k][0])) if i not in idxs],
                         [edges_aug[k][1][i] for i in range(len(edges_aug[k][1])) if i not in idxs]]

    cpg_aug = gen_cpg(nodes_aug, edges_aug)

    return cpg_aug

def gen_aug_cpgs(nodes: torch.Tensor, edges: torch.Tensor, p_node=0.2, p_edge=0.2)->tuple(dgl.DGLGraph, dgl.DGLGraph):
    # node augmentation + edge augmentation
    # node cutoff
    nodes_aug = nodes
    edges_aug = edges
    num = len(nodes)
    if num * p_node > 1:
        nodes_aug[torch.multinomial(torch.arange(0, num), num_samples=math.ceil(num*p_node))] = 0.
    for k in edges:
        if len(edges[k]) and len(edges[k][0]) * p_edge > 1:
            idxs = random.sample(range(len(edges[k][0])), math.ceil(len(edges[k][0]) * p_edge))
            edges_aug[k] = [[edges_aug[k][0][i] for i in range(len(edges_aug[k][0])) if i not in idxs],
                         [edges_aug[k][1][i] for i in range(len(edges_aug[k][1])) if i not in idxs]]
            
    cpg = gen_cpg(nodes, edges)
    cpg_aug = gen_cpg(nodes_aug, edges_aug)

    return cpg, cpg_aug
