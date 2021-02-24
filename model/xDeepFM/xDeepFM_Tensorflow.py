import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from Layers.Layers_Tensorflow import Linear, DNNLayer, CIN
import os
import pickle
from dataset.Criteo import CriteoDatasetTensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class XDeepFM(tf.keras.Model):

    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, cin_layer, cin_direct,
                 dnn_layers=(256, 128), l2_reg=1e-4, dnn_dropout=0, dnn_use_bn=False):
        super(XDeepFM, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.embed_dim = embed_dim
        self.dnn_layers = [len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)] + list(dnn_layers)

        self.l2_reg = l2_reg
        self.embedding = {'embed_' + str(i): Embedding(feat['feat_num'], self.embed_dim,
                                                       embeddings_initializer=glorot_normal(2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear()

        self.dnn = DNNLayer(self.dnn_layers, dnn_dropout, dnn_use_bn, l2_reg)

        self.cin = CIN(cin_layer, cin_direct, l2_reg)

        self.dnn_linear = tf.keras.layers.Dense(input_dim=self.dnn_layers[-1],
                                                units=1,
                                                kernel_initializer=glorot_normal(seed=2020),
                                                kernel_regularizer=l2(self.l2_reg),
                                                use_bias=False)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs

        sparse_embed_list = [self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                             for i in range(sparse_input.shape[1])]

        cin_result = self.cin(tf.stack(sparse_embed_list, axis=1))

        # [Batch, sparse_dim*embed_size]
        sparse_embed_concat = tf.concat(sparse_embed_list, axis=1)

        linear_result = self.linear([dense_input, sparse_embed_concat])

        dnn_result = self.dnn_linear(self.dnn(tf.concat([dense_input, sparse_embed_concat], axis=-1)))

        return tf.nn.sigmoid(linear_result + cin_result + dnn_result)

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

    dataset = CriteoDatasetTensorflow(root_path, 'criteo_1w.lmdb', 32)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    dense_feature, sparse_feature, label = iterator.get_next()
    print(dense_feature)
    print(sparse_feature)
    print(label)

    model = XDeepFM(dense_feature_columns, sparse_feature_columns, embed_dim=8, cin_layer=(128, 128), cin_direct=False,
                    dnn_layers=(256, 128), l2_reg=1e-4, dnn_dropout=0, dnn_use_bn=False)
    output = model([dense_feature, sparse_feature])
    print(output)
    model.summary()
    print(model.trainable_variables)


if __name__ == '__main__':
    test_model()

