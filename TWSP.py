import numpy as np
import scipy.sparse as sp

'''
window_size: the size of the time window that is considered to be related
dataset: dataset to process
path: file path of the dataset 
'''


def generate_plugin_A(window_size, path, n_users, n_items):
    R_dict = {}
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    RI = sp.dok_matrix((n_items, n_items), dtype=np.float32)
    with open(path + '/train.txt') as fp:
        while True:
            info = fp.readline().strip('\n').split(' ')
            if len(info) < 2:
                break
            # if int(info[0]) % 1000 == 0:
            #     print(info[0])
            for i in range(1, len(info)):
                R[int(info[0]), int(info[i])] = 1.
                if info[i] not in R_dict:
                    R_dict[info[i]] = {}
                for j in range(1, len(info)):
                    if i - window_size < j < i + window_size:
                        if info[j] not in R_dict[info[i]]:
                            R_dict[info[i]][info[j]] = 0
                        R_dict[info[i]][info[j]] += 1
        for key, value in R_dict.items():
            for key2, value2 in value.items():
                RI[int(key), int(key2)] = float(value2 / value[key])
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    RI = RI.tolil()
    # prevent memory from overflowing
    for i in range(5):
        adj_mat[int(n_users * i / 5.0): int(n_users * (i + 1.0) / 5), n_users:] = \
            R[int(n_users * i / 5.0): int(n_users * (i + 1.0) / 5)]
        adj_mat[n_users:, int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)] = \
            R[int(n_users * i / 5.0): int(n_users * (i + 1.0) / 5)].T
        adj_mat[n_users + int(n_items * i / 5.0): n_users + int(n_items * (i + 1.0) / 5), n_users:] = \
            RI[int(n_items * i / 5.0): int(n_items * (i + 1.0) / 5)]
    adj_mat = adj_mat.todok()
    return adj_mat
