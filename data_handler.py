import h5py
import scipy.io as scio
import numpy as np

def load_data(path):
    file = h5py.File(path, mode='r')
    images = file['images'][:].astype('float')
    labels = file['LAll'][:]
    tags = file['YAll'][:]

    file.close()

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