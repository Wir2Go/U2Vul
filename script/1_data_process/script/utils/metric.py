import numpy as np
import torch
import torch.nn as nn
import torchmetrics



class Metric(nn.Module):
    def __init__(self, classes=2, average='macro'):
        super().__init__()
        self.acc = 0.0
        self.top1_acc = 0.0
        self.top3_acc = 0.0
        self.top5_acc = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.classes = classes
        # print(self.classes)
        if classes == 2:
            self.task = 'binary'
        elif classes < 512:
            self.task = 'multiclass'
        else:
            self.task = 'multilabel'
        # self.task = 'multiclass' if classes > 2 else 'binary'
        self.epoch = 0

        self.average = average
        self.num_labels = self.classes


    def forward(self, logits, label):
        self.acc = float(torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=1))
        self.precision = float(torchmetrics.functional.precision(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=1))
        self.recall = float(torchmetrics.functional.recall(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=1))
        self.f1_score = float(torchmetrics.functional.f1_score(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=1))
        if self.task != 'binary':
            self.top3_acc = float(torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=3))
            self.top5_acc = float(torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, num_labels=self.num_labels, average=self.average, top_k=5))

    def show_states(self):
        results = f"Epoch: {self.epoch+1}, Accuracy: {self.acc}, Precision: {self.precision}, Recall: {self.recall}, F1-Score: {self.f1_score}"
        if self.task == 'multiclass':
            results += f"\t Top3_Accuracy: {self.top3_acc}, Top5_Accuracy: {self.top5_acc}"

        print(results + '\n', flush=True)

        
def top_k_accuracy(pred:torch.Tensor, label:torch.Tensor, k):
    n = 0
    _, idxs = pred.topk(k, dim=-1)
    # print(idxs.shape)
    # print(idxs)
    for i in range(idxs.shape[0]):
        if torch.any(label[i, idxs[i]]):
            n += 1

    acc = n / len(label)
    dis = []
    for i in range(idxs.shape[0]):
        l_idxs = (label[i] == 1).nonzero(as_tuple=True)[0]
        # print(l_idxs)
        dis.append(min(torch.concat([torch.abs(l_idxs - idxs[i, j]) for j in range(k)], dim=0).view(-1)))
    # print(dis)
    dis = np.mean(torch.stack(dis, dim=0).numpy())

    return acc, dis