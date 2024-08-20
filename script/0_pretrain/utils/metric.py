import torch
import torch.nn as nn
import torchmetrics



class Metric(nn.Module):
    def __init__(self, classes=2, multi_class=False):
        super().__init__()
        self.acc = 0.0
        self.top3_acc = 0.0
        self.top5_acc = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.classes = classes
        self.task = 'multiclass' if multi_class else 'binary'
        self.epoch = 0


    def forward(self, logits, label):
        self.acc = torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, top_k=1)
        self.precision = torchmetrics.functional.precision(logits, label, task=self.task, num_classes=self.classes, top_k=1)
        self.recall = torchmetrics.functional.recall(logits, label, task=self.task, num_classes=self.classes, top_k=1)
        self.f1_score = torchmetrics.functional.f1_score(logits, label, task=self.task, num_classes=self.classes, top_k=1)
        if self.task == 'multiclass':
            self.top3_acc = torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, top_k=3)
            self.top5_acc = torchmetrics.functional.accuracy(logits, label, task=self.task, num_classes=self.classes, top_k=5)

    def show_states(self):
        results = f"Epoch: {self.epoch+1}, Accuracy: {self.acc}, Precision: {self.precision}, Recall: {self.recall}, F1-Score: {self.f1_score}"
        if self.task == 'multiclass':
            results += f"\t Top3_Accuracy: {self.top3_acc}, Top5_Accuracy: {self.top5_acc}"

        print(results + '\n', flush=True)

        
