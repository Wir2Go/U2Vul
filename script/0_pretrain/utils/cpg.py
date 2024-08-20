import sentencepiece as spm
import networkx as nx
import torch, dgl
import numpy as np

def gen_cpg(nodes, edges):
    # print(edges, flush=True)
    cpg_data = {
        ('emb', e.lower(), 'emb'): (torch.as_tensor(edges[e][0]), torch.as_tensor(edges[e][1])) for e in edges if len(edges[e]) > 0
    }
    cpg = dgl.heterograph(cpg_data)
    cpg.nodes['emb'].data['feat'] = torch.as_tensor(np.stack(nodes, axis=0))
    return cpg

def extract_ast(g:nx.classes.digraph.DiGraph, _spm:spm.SentencePieceProcessor)->list:
    order = list(nx.dfs_preorder_nodes(g))
    outs = [g._node[x]['label'] for x in order]
    outs = _spm.EncodeAsIds(outs)
    return outs

def get_edge(g:nx.classes.digraph.DiGraph)->dict[str, list]:
    out_edges = {'AST': [],
                 'CFG': [],
                 'CDG': [],
                 'DDG': []
                 }
    order = list(nx.dfs_preorder_nodes(g))

    raw_edges = g.edges.data()
    for e in raw_edges:
        u, v, a = order.index(e[0]), order.index(e[1]), e[2]['label'].split(':')[0]
        out_edges[a].append((u, v))
    for k in out_edges:
        _x = list(zip(*out_edges[k]))
        _x = [list(n_x) for n_x in _x]
        out_edges[k] = _x
    return out_edges