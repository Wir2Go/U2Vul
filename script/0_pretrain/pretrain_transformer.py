import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.cuda.amp import GradScaler

from model import Transformer
from utils.transformer import AST_Transformer_Dataset, CollateForMLM


LR = 1e-4
batch_size = 32 * 10
num_workers = 15
epochs = 100
warmup_epochs = 10
save_epoch = 5
device = "cuda:0"

# num_sample = 1158283


word_num = 8000
max_len = 512
emb_dim = 512
out_dim = 512

parent_path = os.path.dirname(os.path.dirname(os.path.abspath('.')))
data_path = os.path.join(parent_path, 'data')
model_path = os.path.join(parent_path, 'model/pretrained_transformer')
os.makedirs(model_path, exist_ok=True)

f_train = os.path.join(data_path, 'pretrain_ast.pkl')

word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

collate_mlm = CollateForMLM(word_num=word_num, mlm_prob=0.15, special_tokens=word_dict)

def _collate_fn(data):
    return collate_mlm.dynamic_mask(data)
    
def main():
    model = Transformer.Model(num=word_num, in_dim=emb_dim, out_dim=out_dim)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=LR, params=model.parameters())

    dataset = AST_Transformer_Dataset(f_train, word_num=word_num)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, collate_fn=_collate_fn)

    data_len = len(dataset)
    iter_steps = data_len // batch_size + 1
    warmup_steps = iter_steps * 10
    training_steps = iter_steps * epochs 

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= warmup_steps,
        num_training_steps= training_steps,
    )

    # criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    for i in range(epochs):
        losses = 0.
        for idx, batch_data in enumerate(dataloader):
            ir, src, label = batch_data
            ir, src = ir.to(device), src.to(device)
            label = label.to(device)
            #TODO: pretraining with BERT MLM
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # outs = model(ir, src).cpu()
                # loss = criterion(outs, label)
                loss = model(ir, src, label)
                # loss = criterion(logits.cpu().float().view(-1, word_num), label.view(-1,))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            losses += loss.item()

        print(f"epoch: {i+1:02d}, average loss: {losses/idx:.4f}", flush=True)
        if (i+1) % save_epoch == 0:
            torch.save(model.state_dict(), f"{model_path}/model_{i+1:03d}.pt")

    print("training finish!")
    return 0
if __name__ == "__main__":
    main()



