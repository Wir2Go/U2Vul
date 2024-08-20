import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling

class CPGNN(nn.Module):
    def __init__(self, in_size, out_size, n_steps=2, device='cuda:0'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.ast_gnn = GatedGraphConv(
            in_feats = in_size,
            out_feats = in_size,
            n_steps = n_steps,
            n_etypes=1
        )
        self.fn_1 = nn.Linear(2 * in_size, in_size)
        self.cpg_gnn = GatedGraphConv(
            in_feats = in_size,
            out_feats = in_size,
            n_steps = n_steps,
            n_etypes=3
        )
        self.fn_2 = nn.Linear(2 * in_size, out_size)
        self.device = device
        # self.criterion = nn.MSELoss()

    def forward(self, g: dgl.DGLGraph):
        ast = None
        hiddens = None
        cpg = None

        nodes = g.nodes['emb'].data['feat'].to(self.device)

        if 'ast' in g.etypes:
            ast = dgl.to_homogeneous(dgl.edge_type_subgraph(g, [('emb', 'ast', 'emb')])).to(self.device)
        # cpg = dgl.edge_type_subgraph(g, [('emb', 'cfg', 'emb'),
        #                                                     ('emb', 'cdg', 'emb'),
        #                                                     ('emb', 'ddg', 'emb')])
        
        lt = g.etypes
        lt =  [x for x in g.etypes if x != 'ast']
        # print(g.etypes)
        if len(lt):
            cpg = dgl.edge_type_subgraph(g, lt).to(self.device)
            # print(cpg.etypes)
            edges = torch.zeros(cpg.num_edges(),).to(self.device)
            etypes = cpg.etypes
            for t_etype in etypes:
                # print(g.edges(etype=t_etype, form='eid'))
                edges[g.edges(etype=t_etype, form='eid')] = etypes.index(t_etype)
            cpg = dgl.to_homogeneous(cpg).to(self.device)
        
        # print(cpg.etypes)
    
        # etypes = [g.num_edges(('emb', 'cfg', 'emb')), g.num_edges(('emb', 'cdg', 'emb')), g.num_edges(('emb', 'ddg', 'emb'))]
        # print(etypes)

        
        if ast is not None: 
            hiddens = self.ast_gnn(ast, nodes)
            hiddens = torch.cat([hiddens, nodes], -1)
            hiddens = self.fn_1(hiddens)
        else:
            hiddens = nodes

        if len(lt):
            logits = self.cpg_gnn(cpg, hiddens, edges)
            logits = torch.cat([logits, hiddens], -1)
            logits = self.fn_2(logits)
        else:
            logits = hiddens

        # g.nodes['emb'].data['feat'] = logits

        #TODO: how to pretrain?
        return logits
    
    def pretrain(self, graphs:list[dgl.DGLGraph], graphs_aug:list[dgl.DGLGraph]):
        #TODO: batch -> encode -> MSE Loss -> return Loss

        g = self.forward(dgl.batch(graphs))
        g_aug = self.forward(dgl.batch(graphs_aug))
        x = dgl.unbatch(g)
        x_aug = dgl.unbatch(g_aug)

        loss_1 = self.criterion(x, x_aug)
        loss_2 = 1 / torch.mean(loss_1).squeeze()
        return loss_1 + loss_2

        
class SingleCPGNN(nn.Module):
    def __init__(self, in_size, out_size, n_steps=2, e_type='cfg', device='cuda:0'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.ast_gnn = GatedGraphConv(
            in_feats = in_size,
            out_feats = in_size,
            n_steps = n_steps,
            n_etypes=1
        )
        self.fn_1 = nn.Linear(2 * in_size, in_size)
        self.cpg_gnn = GatedGraphConv(
            in_feats = in_size,
            out_feats = in_size,
            n_steps = n_steps,
            n_etypes=1
        )
        self.fn_2 = nn.Linear(2 * in_size, out_size)

        self.e_type = e_type
        self.device = device
        # self.criterion = nn.MSELoss()

    def forward(self, g: dgl.DGLGraph):
        ast = None
        hiddens = None
        cpg = None

        nodes = g.nodes['emb'].data['feat'].to(self.device)

        if 'ast' in g.etypes:
            ast = dgl.to_homogeneous(dgl.edge_type_subgraph(g, [('emb', 'ast', 'emb')])).to(self.device)
        # cpg = dgl.edge_type_subgraph(g, [('emb', self.e_type, 'emb')])
        # cpg = dgl.edge_type_subgraph(g, [('emb', 'cfg', 'emb'),
        #                                                     ('emb', 'cdg', 'emb'),
        #                                                     ('emb', 'ddg', 'emb')])
        
        lt = g.etypes
        lt =  [x for x in g.etypes if x != 'ast']
        # print(g.etypes)
        if self.e_type in lt:
            cpg = dgl.edge_type_subgraph(g, [self.e_type]).to(self.device)
            # print(cpg.etypes)
            # edges = torch.zeros(cpg.num_edges(),).to(self.device)
            etypes = cpg.etypes
            # for t_etype in etypes:
            #     # print(g.edges(etype=t_etype, form='eid'))
            #     edges[g.edges(etype=t_etype, form='eid')] = etypes.index(t_etype)
            edges = torch.zeros(cpg.num_edges(self.e_type),).to(self.device)
            edges[g.edges(etype=self.e_type, form='eid')] = etypes.index(self.e_type)
            cpg = dgl.to_homogeneous(cpg).to(self.device)
        
        # print(cpg.etypes)
    
        # etypes = [g.num_edges(('emb', 'cfg', 'emb')), g.num_edges(('emb', 'cdg', 'emb')), g.num_edges(('emb', 'ddg', 'emb'))]
        # print(etypes)

        
        if ast is not None: 
            hiddens = self.ast_gnn(ast, nodes)
            hiddens = torch.cat([hiddens, nodes], -1)
            hiddens = self.fn_1(hiddens)
        else:
            hiddens = nodes

        if self.e_type in lt:
            logits = self.cpg_gnn(cpg, hiddens, edges)
            logits = torch.cat([logits, hiddens], -1)
            logits = self.fn_2(logits)
        else:
            logits = hiddens

        # g.nodes['emb'].data['feat'] = logits

        #TODO: how to pretrain?
        return logits
    
    def pretrain(self, graphs:list[dgl.DGLGraph], graphs_aug:list[dgl.DGLGraph]):
        #TODO: batch -> encode -> MSE Loss -> return Loss

        g = self.forward(dgl.batch(graphs))
        g_aug = self.forward(dgl.batch(graphs_aug))
        x = dgl.unbatch(g)
        x_aug = dgl.unbatch(g_aug)

        loss_1 = self.criterion(x, x_aug)
        loss_2 = 1 / torch.mean(loss_1).squeeze()
        return loss_1 + loss_2
    
