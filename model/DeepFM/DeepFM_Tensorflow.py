import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from Layers.Layers_Tensorflow import Linear, DNNLayer
from dataset.Criteo import CriteoDatasetTensorflow
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepFM(tf.keras.Model):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, dnn_units=(256, 256), l2_reg=1e-4,
                 dropout_rate=0.8, use_bn=True):
        """
        DeepFM
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param dnn_units: list. dnn layer size for deep part
        :param dropout_rate: float.
        :param use_bn: bool. use bn or not for dnn layers
        """

        super(DeepFM, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.embed_dim = embed_dim
        self.dnn_layers = [len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)] + list(dnn_units)

        self.l2_reg = l2_reg
        self.embedding = {'embed_' + str(i): Embedding(feat['feat_num'], self.embed_dim,
                                                       embeddings_initializer=glorot_normal(2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear(self.l2_reg)

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, l2_reg=self.l2_reg)

        self.dnn_linear = tf.keras.layers.Dense(input_dim=self.dnn_layers[-1],
                                                units=1,
                                                kernel_initializer=glorot_normal(2020),
                                                kernel_regularizer=l2(self.l2_reg),
                                                use_bias=False)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.shape[1])], axis=1)

        linear_result = self.linear([dense_input, sparse_embed])

        fm_result = 0.5 * tf.reduce_sum(tf.square(tf.reduce_sum(linear_result, axis=1, keepdims=True)) -
                                        tf.reduce_sum(linear_result * linear_result, axis=1, keepdims=True),
                                        axis=1, keepdims=True)

        dnn_result = self.dnn_linear(self.dnn(tf.concat([dense_input, sparse_embed], axis=-1)))

        return tf.nn.sigmoid(linear_result + fm_result + dnn_result)

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


def test_model():
    root_path = '/home/xlm/project/recommendation/dataset/criteo/'
    with open(os.path.join(root_path, 'dense_1w.info'), 'rb') as f:
        dense_feature_columns = pickle.load(f)
    with open(os.path.join(root_path, 'sparse_1w.info'), 'rb') as f:
        sparse_feature_columns = pickle.load(f)
    print('load dense and sparse')

    dataset = CriteoDatasetTensorflow(root_path, 'criteo_1w.lmdb', 8)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    dense_feature, sparse_feature, label = iterator.get_next()
    print(dense_feature)
    print(sparse_feature)
    print(label)

    model = DeepFM(dense_feature_columns, sparse_feature_columns, embed_dim=8)
    # model.summary()
    output = model([dense_feature, sparse_feature])
    print(output)
    # print(model.trainable_variables)


if __name__ == '__main__':
    test_model()

