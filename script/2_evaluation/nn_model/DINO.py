import dgl
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from model_utils import DINO as utils

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
class DiNOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class DiNOWrapper(nn.Module):
    def __init__(self, encoder:nn.Module, mlp:DiNOHead):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp

    # def forward(self, x:list[list[dgl.DGLGraph]]):
    def forward(self, x:dgl.DGLGraph):
        # outs = []
        # for _x in x:
        #     g_out = dgl.unbatch(self.encoder(dgl.batch(_x)))
        #     n_out = torch.cat([torch.mean(g.node.nodes['emb'].data['feat'], dim=0, keepdim=True) for g in g_out], dim=0)
        #     logits = self.mlp(n_out)
        #     outs.append(logits)

        # print(x[0].ndata, x[0].edata)
        # print(x)
        # print(x)
        # x = dgl.batch(x)
        

        # print(edges)
        # edge_types = x.etypes
        # edge_type_tensor = torch.tensor([edge_types.index(et) for et in x.canonical_etypes])
        # print(edge_type_tensor)
        # print(x.edges(etype=['cfg', 'cdg', 'ddg']))
        # ast = dgl.to_homogeneous(dgl.edge_type_subgraph(x, [('emb', 'ast', 'emb')]))
        # print(ast)
        # print(x.nodes['emb'].data)
        x.nodes['emb'].data['feat'] = self.encoder(x)

        
        g_out = dgl.unbatch(x)
        # g_out = dgl.unbatch(self.encoder(dgl.batch(x)))
        n_out = torch.cat([torch.mean(g.nodes['emb'].data['feat'], dim=0, keepdim=True) for g in g_out], dim=0)
        logits = self.mlp(n_out)
        # outs.append(logits)

        return logits

class DiNO(nn.Module):
    def __init__(self, student:DiNOWrapper, teacher:DiNOWrapper, 
                 ncrops = 8, device = 'cpu'):
        super().__init__()
        self.student = student
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.ncrops = ncrops
        self.device = device

    def forward(self, images):
        # in list[list[dgl.DGLGraph (8 local + 2 global)]]
        # images = [[image.to(self.device) for image in image_group] for image_group in images]
        # print(len(images[-2]), len(images[-1]))
        # out (N, M)
        # print(len(images))
        # teacher_output = self.teacher(images[-2] + images[-1])
        # student_output = self.student([image for _images in images for image in _images])
        teacher_output = self.teacher(images[0].to(self.device))
        student_output = self.student(images[1].to(self.device))
        return teacher_output, student_output

    def ema_update(self, m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)