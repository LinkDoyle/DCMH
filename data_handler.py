import h5py
import scipy.io as scio
import numpy as np
import time

def load_data(path):
    start_time = time.time()
    with h5py.File(path, mode='r') as file:
        images = file['images'][:].astype('float')
        labels = file['LAll'][:]
        tags = file['YAll'][:]
    print(f'loaded data {time.time() - start_time:.4f}s')
    labels = np.transpose(labels)
    tags = np.transpose(tags)
    
    print('images.shape', images.shape)
    print('labels.shape', labels.shape)
    print('tags.shape', tags.shape)
    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)