import pickle
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AST_Transformer_Dataset(Dataset):
    def __init__(self, f_in, word_num, max_len=512, pad_val=0, mask_val=3):
        """
        type of ir and src: list consisting of variable-length 1-D numpy array
        """
        with open(f_in, 'rb') as fr:
            data = pickle.load(fr)
        # self.ir, self.src = data
        self.data = data

        self.word_num = word_num
        self.max_len = max_len
        self.pad_val = pad_val
        self.mask_val = mask_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # ir = torch.as_tensor(self.ir[idx], dtype=torch.float32)
        # src = torch.as_tensor(self.src[idx], dtype=torch.float32)
        # ir, src = self.data[idx]
        # src, label = self.mask(src)
        # return (ir, src, label)
        ir, src = self.data[idx]
        
        if self.max_len and min(len(ir), len(src)) > self.max_len:
                idx = random.randint(0, min(len(ir), len(src)) - self.max_len)
                ir, src = ir[idx:], src[idx:]

        ir, src = ir[:self.max_len], src[:self.max_len]
            

        return (torch.as_tensor(ir, dtype=torch.long), torch.as_tensor(src, dtype=torch.long))
    
    def mask(self, x:torch.Tensor):
        i = random.randint(0, len(x)-1)
        p = random.random()
        label = x[i]
        if p < 0.1:
            x[i] = random.randint(self.mask_val+1, self.word_num + self.mask_val -1)
        elif p < 0.9:
            x[i] = self.mask_val

        return x, label
    
class CollateForMLM():
    def __init__(self, word_num, mlm_prob=0.15, special_tokens =None):
        self.mlm_prob = mlm_prob
        
        self.special_tokens = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3} \
            if special_tokens is None else special_tokens
        
        self.last_special_token = list(self.special_tokens.values())[-1]
        self.word_num = word_num
        
    def mask_special_tokens(self, data):
        special_tokens_mask = torch.where(data <= self.last_special_token, 0, 1)
        return special_tokens_mask

    
    def pad_collate_fn(self, data):
        """
        data: is a list of tuples with (example, label, length)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        x_ir, x_src = list(zip(*data))
        x_ir = pad_sequence(x_ir, batch_first=True, padding_value=self.special_tokens['[PAD]'])
        x_src = pad_sequence(x_src, batch_first=True, padding_value=self.special_tokens['[PAD]'])
        return x_ir, x_src

    def dynamic_mask(self, data):
        x_ir, x_src = self.pad_collate_fn(data)
        labels = x_src.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = self.mask_special_tokens(x_src)
        probability_matrix.masked_fill(special_tokens_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # x_src[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        x_src[indices_replaced] = self.special_tokens['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.word_num, labels.shape, dtype=torch.long)
        x_src[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return x_ir.long(), x_src.long(), labels.long()