import os, json

work_path = os.path.join(os.environ['HOME'], 'Work/U2Vul/data/label')

def get_labels(dataset, in_path):
    fl = os.listdir(in_path)
    outs = dict()
    for f in fl:
        f_in = os.path.join(in_path, f)
        with open(f_in, 'r') as fr:
            x = json.load(fr)
        k = x[0]
        if k not in outs:
            outs.setdefault(k, 1)
        else:
            outs[k] = outs[k] + 1

    # print(outs)
    outs = dict(sorted(list(outs.items()), key=lambda x: x[1], reverse=True))
    f_label = os.path.join(work_path, f'{dataset}_class.json')
    with open(f_label, 'w') as fw:
        json.dump(outs, fw)

    return 0

def proc(dataset):
    in_path = os.path.join(work_path, dataset)
    f_label = os.path.join(work_path, f'{dataset}_class.json')
    if dataset == 'juliet':
        get_labels(dataset, in_path)

    with open(f_label, 'r') as fr:
        labels = json.load(fr)

    labels = list(sorted(labels.items(), key=lambda a:a[1], reverse=True))
    labels = [x[0] for x in labels]
    print(f'{dataset}, class: {len(labels)}')

    labels = {labels[i]: i for i in range(len(labels))}

    fl = os.listdir(in_path)
    for f in fl:
        with open(os.path.join(in_path, f), 'r') as fr:
            data = json.load(fr)

        data[0] = labels[data[0]]
        with open(os.path.join(in_path, f), 'w') as fw:
            json.dump(data, fw)

    return 0

def main():
    proc('juliet')
    proc('D2A')
    print('finish')
    return 0

if __name__ == "__main__":
    main()