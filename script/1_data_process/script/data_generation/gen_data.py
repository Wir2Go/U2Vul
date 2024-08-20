import os, sys
import yaml
import random
import json, pickle
import torch

with open(sys.argv[1], 'r') as fr:
    configs = yaml.safe_load(fr)

work_path = os.path.join(os.environ['HOME'], 'Work/U2Vul')
data_path = os.path.join(work_path, 'data')
embedding_path = os.path.join(data_path, 'embedding/evaluation')
label_path = os.path.join(data_path, 'label')
out_path = os.path.join(data_path, 'evaluation')
os.makedirs(out_path, exist_ok=True)

def load_pickle(f_in):
    with open(f_in, 'rb') as fr:
        x = pickle.load(fr)
        return x

def load_json(f_in):
    with open(f_in, 'r') as fr:
        x = json.load(fr)
        return x

def proc_task_binary(task):
    emb_path = os.path.join(embedding_path, task)
    ben_path = os.path.join(emb_path, 'benign')
    vul_list = [x for x in os.listdir(emb_path) if x != 'benign']
    ben_all_fl = os.listdir(ben_path)
    for vul in vul_list:
        vul_path = os.path.join(emb_path, vul)
        vul_fl = os.listdir(vul_path)
        random.shuffle(ben_all_fl)
        ben_fl = ben_all_fl[:len(vul_fl)]

        outs = [(load_pickle(os.path.join(vul_path, f)).cpu(), torch.tensor(1.)) for f in vul_fl]
        outs += [(load_pickle(os.path.join(ben_path, f)).cpu(), torch.tensor(0.)) for f in ben_fl]
        random.shuffle(outs)
        with open(f'{out_path}/{task}_binary_{vul}.pkl', 'wb') as fw:
            pickle.dump(outs, fw)

    return 0

def one_hot(x: list):
    outs = torch.zeros(512).long()
    outs[x] = 1.
    return outs

def proc_task_type(task):
    emb_path = os.path.join(embedding_path, task)
    vul_list = [x for x in os.listdir(emb_path) if x != 'benign']
    for vul in vul_list:
        if vul == 'D2A':
            continue;
        vul_path = os.path.join(emb_path, vul)
        vul_label_path = os.path.join(label_path, vul)
        fl = set([f.split('.')[0] for f in os.listdir(vul_path)]) & set([f.split('.')[0] for f in os.listdir(vul_label_path)])
        print(fl)
        outs_1 = [(load_pickle(os.path.join(vul_path, f'{f}.pkl')).cpu(), torch.as_tensor(load_json(os.path.join(vul_label_path, f'{f}.json'))[0]).long()) for f in fl]
        outs_2 = [(load_pickle(os.path.join(vul_path, f'{f}.pkl')).cpu(), one_hot(load_json(os.path.join(vul_label_path, f'{f}.json'))[1])) for f in fl]
        with open(os.path.join(out_path, f'{task}_cls_{vul}.pkl'), 'wb') as fw:
            pickle.dump(outs_1, fw)
        with open(os.path.join(out_path, f'{task}_loc_{vul}.pkl'), 'wb') as fw:
            pickle.dump(outs_2, fw)

    return 0

def main():
    for t in configs['task']:
        proc_task_binary(t)
        proc_task_type(t)

    print('finish')
    return 0

if __name__ == "__main__":
    main()




