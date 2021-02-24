import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
from Layers.Layers_Tensorflow import Linear, DNNLayer, InteractLayer


class AutoInt(tf.keras.Model):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, att_layer_size=3, att_size=8,
                 head_num=2, use_res=True, dnn_units=(256, 256), dropout_rate=0, use_bn=False, l2_reg=1e-4):
        """
        AutoInt
         :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param att_layer_size: int. attention layer size
        :param att_size: int. attention size
        :param head_num: int.
        :param use_res: bool. use Res connection or not for InteractLayer
        :param dnn_units: list. dnn layer size for deep part
        :param l2_reg:
        :param dropout_rate:
        :param use_bn:
        """
        super(AutoInt, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.embed_dim = embed_dim
        self.att_layer_size = att_layer_size
        self.dnn_layers = [embed_dim * (len(dense_feature_columns) + len(sparse_feature_columns))] + list(dnn_units)

        self.l2_reg = l2_reg

        self.dense_embedding = [self.add_weight(name='dense_embed',
                                                shape=[self.embed_dim],
                                                dtype=tf.float32,
                                                initializer=glorot_normal(seed=2020),
                                                regularizer=l2(self.l2_reg))
                                for _ in range(len(self.dense_feature_columns))]

        self.sparse_embedding = {'embed_' + str(i): Embedding(feat['feat_num'],
                                                              self.embed_dim,
                                                              embeddings_initializer=glorot_normal(seed=2020),
                                                              embeddings_regularizer=l2(self.l2_reg))
                                 for i, feat in enumerate(self.sparse_feature_columns)}

        self.linear = Linear()

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, l2_reg=self.l2_reg)

        self.interact = [InteractLayer(att_size, head_num, use_res=use_res) for _ in range(self.att_layer_size)]

        self.dense = tf.keras.layers.Dense(units=1,
                                           kernel_initializer=glorot_normal(seed=2020),
                                           kernel_regularizer=l2(self.l2_reg),
                                           use_bias=False)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        sparse_embed = tf.stack([self.sparse_embedding['embed_{}'.format(i)](sparse_input[:, i])
                                 for i in range(sparse_input.shape[1])], axis=1)

        dense_embed = tf.stack([tf.tensordot(dense_input[:, i], self.dense_embedding[i], axes=0)
                                for i in range(dense_input.shape[1])], axis=1)

        linear_result = self.linear([dense_input, Flatten()(sparse_embed)])

        dnn_output = self.dnn(tf.concat([Flatten()(dense_embed), Flatten()(sparse_embed)], axis=-1))

        interact_input = tf.concat([dense_embed, sparse_embed], axis=1)

        for i in range(self.att_layer_size):
            interact_input = self.interact[i](interact_input)

        # [Batch, (dense_dim + sparse_dim)*att_size*head_num]
        interact_output = Flatten()(interact_input)

        autoint_result = self.dense(tf.concat([dnn_output, interact_output], axis=1))

        return tf.nn.sigmoid(linear_result + autoint_result)

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()



