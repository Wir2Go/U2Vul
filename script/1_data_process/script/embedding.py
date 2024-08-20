import os, sys

import torch
import torch.nn as nn

import numpy as np
import sentencepiece as spm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nn_model import Transformer, GNN, DINO
from model_utils.NC import *
from utils.cpg import gen_cpg, extract_ast, get_edge

word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

work_path = os.path.join(os.environ['HOME'], 'Work/Water_Vul/U2Vul')

model_path = os.path.join(work_path, "model")
sp_path = os.path.join(model_path, 'tokenization')

transformer_path = os.path.join(model_path, "transformer.pt")
GGNN_path = os.path.join(model_path, "ggnn.pt")

data_path = os.path.join(work_path, 'data')
in_path = os.path.join(data_path, 'cpg')

out_path = os.path.join(data_path, 'embedding/evaluation')
os.makedirs(out_path, exist_ok=True)

class Bunch:
    def __init__(self, **entries):
        self.__dict__.update(entries)

tokenization_args = {

}
transformer_args = {
    'word_num': 8000,
    'in_dim': 512,
    'out_dim': 512,
}
ggnn_args = {
    'word_num': 8000,
    'max_len': 512,
    'emb_dim': 512,
    'out_dim': 1024,
}

device = 'cuda:0'
# device = 'cpu'

sp_ir = spm.SentencePieceProcessor(os.path.join(sp_path, 'ir_.model'))
sp_src = spm.SentencePieceProcessor(os.path.join(sp_path, 'src_.model'))

def initial_model(model:nn.Module, model_path:str, **kwargs)->nn.Module:
    model = model(kwargs)
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model.eval()
    return model

def initial_transformer():
    model = Transformer.Model(num=transformer_args['word_num'], 
                              in_dim=transformer_args['in_dim'], 
                              out_dim=transformer_args['out_dim'])
    model.load_state_dict(torch.load(transformer_path))
    model = model.to(device)
    model.eval()
    return model

def initial_ggnn():
    model = GNN.CPGNN(in_size=ggnn_args['emb_dim'], out_size=ggnn_args['emb_dim'], device=device)
    model.load_state_dict(torch.load(GGNN_path))
    model = model.to(device)
    model.eval()
    return model

def emb_transformer(ir, src, model:Transformer.Model):
    in_ir = list()
    in_src = list()
    idxs = list()
    outs = list()
    count = 0

    for x in ir:
        in_ir.extend(x + [1])
    in_ir.append(2)
    for x in src:
        in_src.extend(x + [1])
        idxs.append(list(range(count, count+len(x))))
        count += len(x)
    in_src.append(2)

    ir, src = torch.as_tensor(in_ir).unsqueeze(0).long().to(device),  torch.as_tensor(in_src).unsqueeze(0).long().to(device)
    if ir.shape[-1] > transformer_args['word_num'] or src.shape[-1] > transformer_args['word_num']:
        return 0

    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        embs = model.encode(ir, src).squeeze().numpy(force=True)
        for x in idxs:
            # print(x)
            if len(x) > 1:
                outs.append(np.mean(embs[x], axis=0))
            else:
                outs.append(embs[x])
    return outs

def proc(f_in, f_out, transformer:Transformer.Model, gnn: GNN.CPGNN,):

    if os.path.exists(f_out):
        return 0

    try:
        in_data = load_pickle(f_in)
    except:
        return 0
    ir, src = in_data

    # get ast, CPG, encode with sentencepiece, keep mapping
    ir_ast = extract_ast(ir, sp_ir)
    src_ast = extract_ast(src, sp_src)
    src_edge = get_edge(src)
            

    with torch.no_grad():
        # embedding with transformer
        embs = emb_transformer(ir_ast, src_ast, transformer)
        if embs == 0:
            return 0
        cpg = gen_cpg(embs, src_edge)
        # embedding with GGNN
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out_x = gnn(cpg)

    save_pickle(out_x, f_out)
    return 0






def main():
    transformer = initial_transformer()
    ggnn = initial_ggnn()


    subs = os.listdir(in_path)
    for sub in subs:
        fl = os.listdir(os.path.join(in_path, sub))
        in_dir = os.path.join(in_path, sub)
        out_dir = os.path.join(out_path, sub)
        os.makedirs(out_dir, exist_ok=True)
        print(out_dir)
        # fl = [x for x in fl if '.pkl' in x]

        for f in fl:
            f_in = os.path.join(in_dir, f)
            f_out = os.path.join(out_dir, f)
            proc(f_in, f_out, transformer, ggnn)

    print('finish')
    return 0

if __name__ == "__main__":
    main()
    
