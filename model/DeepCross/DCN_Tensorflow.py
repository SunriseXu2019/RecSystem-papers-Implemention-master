import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from Layers.Layers_Tensorflow import Linear, DNNLayer, Cross
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DCN(tf.keras.Model):

    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, cross_num, dnn_units=(256, 256),
                 dropout_rate=0, use_bn=False, l2_reg=1e-4):
        """
        Deep & Cross
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param cross_num: int.
        :param dnn_units: list. dnn layer size for deep part
        :param dropout_rate:
        :param use_bn:
        :param l2_reg:
        """
        super(DCN, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.embed_dim = embed_dim
        self.field_dim = len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)
        self.dnn_layers = [self.field_dim] + list(dnn_units)
        self.l2_reg = l2_reg
        self.embedding = {'embed_' + str(i): Embedding(feat['feat_num'], self.embed_dim,
                                                       embeddings_initializer=glorot_normal(2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear(self.l2_reg)

        self.cross = Cross(cross_num, self.field_dim, l2_reg=1e-4)

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, l2_reg=self.l2_reg)

        self.dnn_cross_linear = tf.keras.layers.Dense(input_dim=self.field_dim + self.dnn_layers[-1],
                                                      units=1,
                                                      kernel_initializer=glorot_normal(seed=2020),
                                                      kernel_regularizer=l2(self.l2_reg),
                                                      use_bias=False)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.shape[1])], axis=1)

        linear_result = self.linear([dense_input, sparse_embed])

        # [Batch, dense + sparse * embed_dim]
        dense_sparse_concat = tf.concat([dense_input, sparse_embed], axis=1)

        # [Batch, dense + sparse * embed_dim]
        cross_output = self.cross(dense_sparse_concat)

        # [Batch, dnn_out_size]
        dnn_output = self.dnn(dense_sparse_concat)

        feature_cross_result = self.dnn_cross_linear(tf.concat([cross_output, dnn_output], axis=1))

        return tf.nn.sigmoid(linear_result + feature_cross_result)

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


