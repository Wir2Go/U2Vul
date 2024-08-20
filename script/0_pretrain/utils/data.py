import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class My_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
def My_collate(data):
    x, label = tuple(zip(*data))
    # print(label)
    x = pad_sequence(list(x), batch_first=True).float()
    label = torch.as_tensor(label).long()

    return x, label

        