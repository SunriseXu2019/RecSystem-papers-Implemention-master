import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from dataset.Criteo import CriteoDatasetTensorflow
import os
import pickle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Linear(tf.keras.layers.Layer):
    def __init__(self, field_dim, l2_reg=1e-4):
        """
        Linear model for FFM model
        :param field_dim: a list of positive numbers, dim of each feature field
        :param l2_reg:
        """
        super(Linear, self).__init__()
        self.l2_reg = l2_reg
        self.field_dim = field_dim
        self.bias = self.add_weight(name='weight_bias',
                                    shape=1,
                                    initializer=Zeros(),
                                    trainable=True)
        self.weight_sparse = Embedding(sum(self.field_dim), 1,
                                       embeddings_initializer=glorot_normal(2020),
                                       embeddings_regularizer=l2(self.l2_reg))

    def build(self, input_shape):
        self.weight_dense = self.add_weight(name='weight_dense',
                                            shape=[int(input_shape[0][-1]), 1],
                                            initializer=glorot_normal(2020),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        linear_dense = tf.tensordot(dense_input, self.weight_dense, axes=(1, 0))
        linear_sparse = tf.reduce_sum(self.weight_sparse(sparse_input), axis=1, keepdims=True)
        return self.bias + linear_dense + linear_sparse


class FFMLayer(tf.keras.Model):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, l2_reg=1e-4):
        super(FFMLayer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.embed_dim = embed_dim
        self.field_dim = [feat['feat_num'] for feat in self.sparse_feature_columns]

        self.offset = tf.convert_to_tensor(np.array((0, *np.cumsum(self.field_dim)[:-1]), dtype=np.int32))
        self.l2_reg = l2_reg
        self.embedding = {'embed_' + str(i): Embedding(sum(self.field_dim), self.embed_dim,
                                                       embeddings_initializer=glorot_normal(2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear(self.field_dim, self.l2_reg)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs

        # 构建one-hot的索引形式，self.offset保存sparse info中各个特征域的维度和
        sparse_input = sparse_input + self.offset
        linear_result = self.linear([dense_input, sparse_input])

        sparse_embed = [self.embedding['embed_{}'.format(i)](sparse_input)
                        for i in range(sparse_input.shape[1])]
        feature_cross = []
        for i in range(len(self.field_dim) - 1):
            for j in range(i + 1, len(self.field_dim)):
                feature_cross.append(tf.reduce_sum(sparse_embed[j][:, i] * sparse_embed[i][:, j],
                                                   axis=1, keepdims=True))

        feature_cross_result = tf.reduce_sum(tf.concat(feature_cross, axis=1), axis=1, keepdims=True)

        return tf.sigmoid(linear_result + feature_cross_result)

    def summary(self, **kwargs):
        dense_input = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_input = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_input, sparse_input], outputs=self.call([dense_input, sparse_input])).summary()


def test_model():
    root_path = '/home/xlm/project/recommendation/dataset/criteo/'
    with open(os.path.join(root_path, 'dense_1w.info'), 'rb') as f:
        dense_feature_columns = pickle.load(f)
    with open(os.path.join(root_path, 'sparse_1w.info'), 'rb') as f:
        sparse_feature_columns = pickle.load(f)
    print('load dense and sparse')

    dataset = CriteoDatasetTensorflow(root_path, 'criteo_1w.lmdb', batch_size=8)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    dense_feature, sparse_feature, label = iterator.get_next()
    print(dense_feature)
    print(sparse_feature)
    print(label)

    model = FFMLayer(dense_feature_columns, sparse_feature_columns, embed_dim=8)
    model.summary()
    output = model([dense_feature, sparse_feature])
    print(output)
    print(model.trainable_variables)


if __name__ == '__main__':
    test_model()




