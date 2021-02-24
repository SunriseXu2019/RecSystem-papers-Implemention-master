import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.keras.losses import binary_crossentropy
from Layers.Layers_Tensorflow import Linear


class AFM(tf.keras.Model):

    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, att_size, dropout_rate, l2_reg=1e-4):
        """
        AFM
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param att_size: int. attention size
        :param dropout_rate: float. dropout rate
        :param l2_reg:
        """
        super(AFM, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.embed_dim = embed_dim
        self.att_size = att_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.linear = Linear(self.l2_reg)
        self.embedding = {'embed_' + str(i): Embedding(feat['feat_num'], self.embed_dim,
                                                       embeddings_initializer=glorot_normal(2020),
                                                       embeddings_regularizer=l2(self.l2_reg))
                          for i, feat in enumerate(self.sparse_feature_columns)}

    def build(self, input_shape):

        self.att_w = self.add_weight(name='att_w',
                                     shape=[self.embed_dim, self.att_size],
                                     initializer=glorot_normal(2020),
                                     regularizer=l2(self.l2_reg),
                                     trainable=True)

        self.att_b = self.add_weight(name='att_b',
                                     shape=[self.att_size],
                                     initializer=glorot_normal(2020),
                                     regularizer=l2(self.l2_reg),
                                     trainable=True)

        self.att_h = self.add_weight(name='att_h',
                                     shape=[self.att_size, 1],
                                     initializer=glorot_normal(2020),
                                     regularizer=l2(self.l2_reg),
                                     trainable=True)

        self.att_p = self.add_weight(name='att_p',
                                     shape=[self.embed_dim, 1],
                                     initializer=glorot_normal(2020),
                                     regularizer=l2(self.l2_reg),
                                     trainable=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super(AFM, self).build(input_shape)

    def call(self, inputs, **kwargs):

        dense_input, sparse_input = inputs

        sparse_embed_list = [self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                             for i in range(sparse_input.shape[1])]

        # [Batch, field_dim*embed_size]
        sparse_embed_concat = tf.concat(sparse_embed_list, axis=1)

        # [Batch, field_dim, embed_size]
        sparse_embed_stack = tf.stack(sparse_embed_list, axis=1)

        linear_result = self.linear([dense_input, sparse_embed_concat])

        element_wise_product_list = []
        for i in range(sparse_input.shape[1] - 1):
            for j in range(i + 1, sparse_input.shape[1]):
                element_wise_product_list.append(tf.multiply(sparse_embed_stack[:, i, :], sparse_embed_stack[:, j, :]))

        # [Batch, field_num*(field_num - 1)/2, embed_size]
        element_wise_product = tf.stack(element_wise_product_list, axis=1)

        # [Batch, field_num*(field_num - 1)/2, att_size]
        att_wx_b = tf.nn.relu(tf.add(tf.tensordot(element_wise_product, self.att_w, axes=(-1, 0)), self.att_b))

        # [Batch, field_num*(field_num - 1)/2, 1]
        normalized_att_score = tf.nn.softmax(tf.tensordot(att_wx_b, self.att_h, axes=(-1, 0)), axis=1)

        # [Batch, field_num*(field_num - 1)/2, embed_size]
        att_output = tf.reduce_sum(normalized_att_score * element_wise_product, axis=1)

        att_output = self.dropout(att_output)

        # [batch, 1]
        afm_result = tf.tensordot(att_output, self.att_p, axes=(-1, 0))

        return tf.sigmoid(afm_result + linear_result)

    def summary(self, **kwargs):
        dense_input = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_input = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_input, sparse_input], outputs=self.call([dense_input, sparse_input])).summary()


