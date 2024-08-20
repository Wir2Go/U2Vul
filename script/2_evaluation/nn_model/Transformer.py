import torch
import torch.nn as nn
import torch.nn.functional as F



class Embedding_Layer(nn.Module):
    def __init__(self, num, emb_dim, padding_idx=0, max_norm=1):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=num, embedding_dim=emb_dim, 
                                      padding_idx=padding_idx, max_norm=max_norm)
        
        self.pos_emb = nn.Embedding(num_embeddings=num, embedding_dim=emb_dim, 
                                padding_idx=padding_idx, max_norm=max_norm)
        

    def forward(self, x: torch.Tensor):
        pos = torch.arange(start=0, end=x.shape[-1], dtype=torch.long, device=x.device)
        # print(x.shape, pos.shape)
        emb_x = self.token_emb(x)
        emb_pos = self.pos_emb(pos)
        return emb_x + emb_pos
        
class Model(nn.Module):
    def __init__(self, num, in_dim, out_dim):
        super().__init__()
        encode_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=8, batch_first=True)
        decode_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=6)
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=6)
        self.encode_embedding = Embedding_Layer(num, in_dim)
        self.decode_embedding = Embedding_Layer(num, in_dim)

        self.ln = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.num = num

    """
    ir: padded ir token sequences
    src: padded src token sqeuences
    outs: features of src embedded with the features extracted from ir.
    """
    def forward(self, ir, src, label):
        x_ir = self.encode_embedding(ir)
        x_src = self.decode_embedding(src)
        # label = self.decode_embedding.token_emb(label)
        logits = self.encoder(x_ir)
        outs = self.decoder(x_src, logits)
        logits = self.ln(outs)
        # loss = F.cross_entropy(logits.view(-1, self.num), label.view(-1,))
        loss = self.criterion(logits.view(-1, self.num), label.view(-1,))
        return loss

    def encode(self, ir, src):
        with torch.no_grad():
            x_ir = self.encode_embedding(ir)
            x_src = self.decode_embedding(src)
            logits = self.encoder(x_ir)
            outs = self.decoder(x_src, logits)
        return outs

class Single_Model(nn.Module):
    def __init__(self, num, in_dim, out_dim):
        super().__init__()
        encode_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=6)
        self.encode_embedding = Embedding_Layer(num, in_dim)

        self.ln = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.num = num

    """
    ir: padded ir token sequences
    src: padded src token sqeuences
    outs: features of src embedded with the features extracted from ir.
    """
    def forward(self, src, label):
        x_src = self.encode_embedding(src)
        # label = self.decode_embedding.token_emb(label)
        outs = self.encoder(x_src)
        logits = self.ln(outs)
        # loss = F.cross_entropy(logits.view(-1, self.num), label.view(-1,))
        loss = self.criterion(logits.view(-1, self.num), label.view(-1,))
        return loss
        
    def encode(self, src):
        with torch.no_grad():
            x_src = self.encode_embedding(src)
            outs = self.encoder(x_src)
        return outs
        

        
