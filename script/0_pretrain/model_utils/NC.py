import pickle

def load_pickle(f_path):
    with open(f_path, 'rb') as fr:
        data = pickle.load(fr)
    return data

def save_pickle(data, f_path):
    with open(f_path, 'wb') as fw:
        pickle.dump(data, fw, protocol=5)
    return 0