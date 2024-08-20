import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler
from sklearn.model_selection import KFold

from uuid import uuid4

import yaml
import json, pickle

with open(sys.argv[1], 'r') as fr:
    configs = yaml.safe_load(fr)

# work_path = os.path.join(os.environ['HOME'], 'Work/U2Vul/script/evaluation')
work_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(work_path)
from nn_model.DownStream import *
from utils.metric import Metric
from utils.data import My_Dataset, My_collate

log_path = os.path.join(work_path, 'log')
os.makedirs(log_path, exist_ok=True)

data_path = os.path.join(work_path, 'data/evaluation')


torch.set_num_threads(torch.multiprocessing.cpu_count() // 2)

batch_size = 32
LR = 3e-4 * batch_size / 32
epochs = 100
k_fold = 5

in_dim = 512
out_dim = configs['class']

device = configs['device']

def init_config(cur_type, task, dataset, cur_model):
    global f_params, out_dim, f_in, f_out, cur_task
    cur_task = task
    f_in = os.path.join(data_path, f'{cur_type}_{task}_{dataset}.pkl')
    if task == 'binary':
        f_out = os.path.join(log_path, f'{cur_type}_{task}_{dataset}_{cur_model}.json')
    else:
        f_out = os.path.join(log_path, f'{cur_type}_{task}_{dataset}_{configs["average"]}_{cur_model}.json')

    print(f_out)
    f_params = f'tmp_models/{uuid4()}.pt'
    if task != 'cls':
        out_dim = configs['class'][task]
    else:
        out_dim = configs['class'][task][dataset]
    return 0


def train(model, train_loader: DataLoader, criterion, optimizer):
    model.train()
    # scaler = GradScaler()
    losses = 0.0
    for step, batch_data in enumerate(train_loader):
        x, label = batch_data
        x = x.to(device)
        # print(x.shape)
        optimizer.zero_grad()
        logits = model(x).cpu()
        
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / (step + 1) 

def eval(model, test_loader):
    k_metric = Metric(classes=out_dim, average=configs['average'])
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for step, batch_data in enumerate(test_loader):
            x, label = batch_data
            x = x.to(device)
            logits = model(x).cpu().view(-1, out_dim)
            all_logits.append(logits.cpu())
            if cur_task == 'loc':
                all_labels.append(label.view(-1, out_dim))
            else:
                all_labels.append(label.view(-1, 1))
                

        all_logits = torch.concat(all_logits)
        all_labels = torch.concat(all_labels)
        # print(all_labels.shape)
        if cur_task != 'loc':
            all_labels = all_labels.squeeze()
            if cur_task == 'binary':
                _, all_logits = all_logits.cpu().topk(1, dim=1)
            all_logits = all_logits.squeeze()
        else:
            all_logits = np.round(all_logits)

        k_metric(all_logits, all_labels)
        # k_metric.show_states()
        _val = (k_metric.acc, k_metric.precision, k_metric.recall, k_metric.f1_score, k_metric.top3_acc, k_metric.top5_acc)
        return _val
    
def proc(model:nn.Module, criterion, optimizer, data):
    global k_fold
    dataset = My_Dataset(data)
    kfold = KFold(k_fold)
    final_results = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Start the {fold}-th evaluatoin.\n========================================================================\n')
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(
                        dataset, 
                        batch_size=batch_size, collate_fn=My_collate, sampler=train_subsampler)
        testloader = DataLoader(
                        dataset,
                        batch_size=batch_size, collate_fn=My_collate, sampler=test_subsampler)
        
        init_params = torch.load(f_params)
        model.load_state_dict(init_params['model'])
        optimizer.load_state_dict(init_params['optimizer'])
        
        
        model = model.to(device)

        for n in range(epochs):
            loss = train(model, trainloader, criterion, optimizer)
            acc, prec, recall, f1, top3, top5 = eval(model, testloader)

            results = f"Epoch: {n+1:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
            if cur_task != 'binary':
                results += f', Top3-Accuracy: {top3:.4f}, Top5-Accuracy: {top5:.4f}'
            results += '\n'
            print(results)

        final_results.append((acc, prec, recall, f1, top3, top5))
        print(f'Finish the {fold}-th evaluatoin.\n------------------------------------------------------------------------\n')
        # break since we only need 1 validation for figure depiction.
        break;
    return 0    


def main():
    for cur_type in configs['type']:
        for cur_dataset in configs['dataset']:
            for task in configs['task']:
                for cur_model in configs['model']:
                    init_config(cur_type, task, cur_dataset, cur_model)
                    f_model = f'tmp_models/{task}_{cur_dataset}.pt'

                    with open(f_in, 'rb') as fr:
                        data = pickle.load(fr)
                    model = globals()[cur_model](in_dim=in_dim, out_dim=out_dim)
                    # model = Corse(in_dim=in_dim, out_dim=out_dim)
                    # model = TextCNN(in_dim=in_dim, out_dim=out_dim)
                    if cur_task == 'loc':
                        criterion = nn.MultiLabelSoftMarginLoss()
                    else:
                        criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
                    # optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
                    # optimizer = pytorch_optimizer.AdaBound(params=model.parameters(), lr=LR, final_lr=3e-4)

                    init_params = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(init_params, f_params)

                    proc(model, criterion, optimizer, data)
                    os.remove(f_params)

                    # save model params
                    model_params = model.state_dict()
                    for k, v in model_params.items():
                        model_params[k] = v.cpu()
                    torch.save(model_params, f_model)
        print('finish')
        return 0


if __name__ == "__main__":
    main()
