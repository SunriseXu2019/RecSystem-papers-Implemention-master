import struct
import lmdb
import numpy as np
import torch.utils.data
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def ReadCriteoDataset(data_path, cache_path):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = ['label'] + dense_features + sparse_features

    data_df = pd.read_csv(os.path.join(data_path, 'train.txt'), sep='\t', header=None, names=features)

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # 1.Label Encoding for sparse features and do simple Transformation for dense features
    for feat in sparse_features:
        data_df[feat] = LabelEncoder().fit_transform(data_df[feat])
    data_df[dense_features] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_df[dense_features])

    # 2.count #unique features for each sparse field and record dense feature field name
    dense_feature_columns = [{'name': feat} for feat in dense_features]
    sparse_feature_columns = [{'name': feat, 'feat_num': len(data_df[feat].unique())}
                              for feat in sparse_features]

    with open(os.path.join(data_path, 'dense.info'), 'wb') as f:
        pickle.dump(dense_feature_columns, f)
        f.close()

    with open(os.path.join(data_path, 'sparse.info'), 'wb') as f:
        pickle.dump(sparse_feature_columns, f)
        f.close()

    if cache_path is None:
        raise ValueError('create cache: failed: cache path is None')

    # create lmdb dataset for criteo dataset
    with lmdb.open(os.path.join(cache_path, 'train.lmdb'), map_size=int(1e11)) as env:
        with env.begin(write=True) as txn:
            for index in tqdm(range(0, int(4.1e7))):
                line = np.array(data_df.loc[index, :])
                txn.put(struct.pack('>I', index), line.tobytes())
        print('success put train data in lmdb')

    with lmdb.open(os.path.join(cache_path, 'val.lmdb'), map_size=int(1e11)) as env:
        with env.begin(write=True) as txn:
            for index in tqdm(range(int(4.1e7), len(data_df))):
                index -= int(4.1e7)
                line = np.array(data_df.loc[index, :])
                txn.put(struct.pack('>I', index), line.tobytes())
        print('success put val data in lmdb')


class CriteoDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, root_path, filename):
        self.env = lmdb.open(os.path.join(root_path, filename), create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(txn.get(struct.pack('>I', index)))
        label = np_array[0]
        dense_feature = np_array[1:14]
        sparse_feature = np_array[14:]
        return dense_feature, sparse_feature, label

    def __len__(self):
        return self.length


def CriteoDatasetTensorflow(root_path, filename, batch_size):
    env = lmdb.open(os.path.join(root_path, filename), create=False, lock=False, readonly=True)
    with env.begin(write=False) as txn:
        length = txn.stat()['entries'] - 1

    dense_feature = np.zeros((length, 13), dtype=np.float32)
    sparse_feature = np.zeros((length, 26), dtype=np.int32)
    label = np.zeros((length, 1), dtype=np.int32)

    with env.begin(write=False) as txn:
        for index in range(length):
            np_array = np.frombuffer(txn.get(struct.pack('>I', index)))
            label[index] = np_array[0]
            dense_feature[index] = np_array[1:14]
            sparse_feature[index] = np_array[14:]

    dataset = tf.data.Dataset.from_tensor_slices((dense_feature, sparse_feature, label))
    dataset = dataset.shuffle(buffer_size=20480).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    data_path = '/home/xlm/project/recommendation/dataset/criteo/'
    ReadCriteoDataset(data_path, data_path)