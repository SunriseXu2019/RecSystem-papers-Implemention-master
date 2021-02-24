import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from Layers.Layers_Tensorflow import Linear


class FMLayer(tf.keras.Model):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, l2_reg=1e-4):
        super(FMLayer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.embed_dim = embed_dim
        self.l2_reg = l2_reg

        self.embedding = {'embed_' + str(i): Embedding(feat['feat_num'], self.embed_dim,
                                                       embeddings_initializer=glorot_normal(seed=2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear(self.l2_reg)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.shape[1])], axis=1)

        linear_result = self.linear([dense_input, sparse_embed])

        features_cross_result = 0.5 * tf.reduce_sum(tf.square(tf.reduce_sum(sparse_embed, axis=1, keepdims=True)) -
                                                    tf.reduce_sum(sparse_embed * sparse_embed, axis=1, keepdims=True),
                                                    axis=1, keepdims=True)

        return tf.sigmoid(linear_result + features_cross_result)

    def summary(self, **kwargs):
        dense_input = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_input = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_input, sparse_input], outputs=self.call([dense_input, sparse_input])).summary()




