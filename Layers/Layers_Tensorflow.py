import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Zeros, glorot_normal, glorot_uniform, TruncatedNormal


class Linear(tf.keras.layers.Layer):
    def __init__(self, l2_reg=1e-4):
        super(Linear, self).__init__()
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.bias = self.add_weight(name='weight_bias',
                                    shape=1,
                                    initializer=Zeros(),
                                    trainable=True)
        self.weight_dense = self.add_weight(name='weight_dense',
                                            shape=[int(input_shape[0][-1]), 1],
                                            initializer=glorot_normal(2020),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True)

        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dense_input, sparse_input = inputs
        linear_dense = tf.tensordot(dense_input, self.weight_dense, axes=(1, 0))
        linear_sparse = tf.reduce_sum(sparse_input, axis=-1, keepdims=True)
        return self.bias + linear_dense + linear_sparse


class DNNLayer(tf.keras.layers.Layer):
    def __init__(self, dnn_layers, dropout_rate, use_bn, l2_reg=1e-4):
        """
        DNN part for DeepFM, xDeepFM, Deep&Cross
        :param dnn_layers: list of positive integer, the layer number and units in each layer
        :param dropout_rate:
        :param use_bn:
        :param l2_reg:
        """
        super(DNNLayer, self).__init__()
        self.dnn_layers = dnn_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        self.weight = [self.add_weight(name='weight' + str(i),
                                       shape=(self.dnn_layers[i], self.dnn_layers[i + 1]),
                                       initializer=glorot_normal(seed=2020),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(len(self.dnn_layers) - 1)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.dnn_layers[i + 1],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.dnn_layers) - 1)]
        if self.use_bn:
            self.bn = [tf.keras.layers.BatchNormalization() for _ in range(len(self.dnn_layers) - 1)]

        self.dropout = [tf.keras.layers.Dropout(self.dropout_rate, seed=2020 + i)
                        for i in range(len(self.dnn_layers) - 1)]

    def call(self, inputs, **kwargs):
        """
        :param inputs: [Batch, dnn_dim*embed_size]
        :param kwargs:
        :return: [Batch, 1]
        """
        dnn_input = inputs

        for i in range(len(self.weight)):
            x = tf.nn.bias_add(tf.tensordot(dnn_input, self.weight[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                x = self.bn[i](x)

            x = tf.keras.layers.ReLU()(x)

            x = self.dropout[i](x)
            dnn_input = x

        dnn_output = dnn_input
        return dnn_output


class Cross(tf.keras.layers.Layer):
    def __init__(self, cross_num, field_dim, l2_reg=1e-4):
        """
        Feature Cross part for Deep&Cross
        :param cross_num: cross layer num
        :param field_dim: feature field dim
        :param l2_reg:
        """
        super(Cross, self).__init__()
        self.cross_num = cross_num
        self.field_dim = field_dim
        self.l2_reg = l2_reg
        self.weight = [self.add_weight(name='weight' + str(i),
                                       shape=(self.field_dim, 1),
                                       initializer=glorot_normal(seed=2020),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.cross_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.field_dim,),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.cross_num)]

    def call(self, inputs, **kwargs):
        x_0 = inputs
        x_l = x_0
        for i in range(self.cross_num):
            x_l += tf.multiply(x_0, tf.tensordot(x_l, self.weight[i], axes=(1, 0))) + self.bias[i]
        return x_l


class CIN(tf.keras.layers.Layer):
    def __init__(self, cin_layer, direct, l2_reg):
        """
        CIN layers for xDeepFM
        :param cin_layer: a list, cin layers
        :param direct: direct or no-direct
        :param l2_reg:
        """
        super(CIN, self).__init__()
        self.cin_layer = cin_layer
        self.direct = direct
        self.l2_reg = l2_reg
        if self.direct:
            self.output_size = sum([layer for layer in self.cin_layer])
        else:
            self.output_size = sum([int(layer / 2) for layer in self.cin_layer])

        self.conv_layers = [tf.keras.layers.Conv1D(filters=self.cin_layer[i],
                                                   kernel_size=1,
                                                   kernel_initializer=glorot_uniform(seed=2020),
                                                   bias_initializer='zeros',
                                                   kernel_regularizer=l2(self.l2_reg),
                                                   bias_regularizer=l2(self.l2_reg),
                                                   activation='relu')
                            for i in range(len(self.cin_layer))]

        self.cin_linear = tf.keras.layers.Dense(input_dim=self.output_size,
                                                units=1,
                                                kernel_initializer=glorot_normal(seed=2020),
                                                kernel_regularizer=l2(self.l2_reg),
                                                use_bias=False)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [Batch, sparse_dim, embed_size]
        :return:
        """
        cin_layers = [inputs]
        cin_result = []

        for i, layer_size in enumerate(self.cin_layer):

            # [Batch, Hk, H0(sparse_dim), embed_size]
            x = tf.einsum('ihk, imk -> ihmk', cin_layers[-1], cin_layers[0])

            # [Batch, Hk*H0, embed_size]
            x = tf.reshape(x, [-1, cin_layers[-1].shape[1] * cin_layers[0].shape[1], inputs.shape[-1]])

            # [Batch, embed_size, Hk+1]
            # tf.keras.layers.Conv1D 是纵向的卷积层 而torch.nn.Conv1D是横向卷积层，tf需要进行transpose操作
            x = self.conv_layers[i](tf.transpose(x, perm=[0, 2, 1]))

            # [Batch, Hk+1, embed_size]
            curr_out = tf.transpose(x, perm=[0, 2, 1])

            if self.direct:
                cin_layers.append(curr_out)
                cin_result.append(curr_out)
            else:
                if i <= len(self.cin_layer) - 1:
                    a, b = tf.split(curr_out, [int(layer_size / 2)] * 2, 1)
                    cin_result.append(a)
                    cin_layers.append(b)
                else:
                    cin_result.append(curr_out)

        # [Batch, self.output_size]
        cin_result = tf.reduce_sum(tf.concat(cin_result, axis=1), axis=-1)
        return self.cin_linear(cin_result)


class InteractLayer(tf.keras.layers.Layer):
    def __init__(self, att_size, head_num, use_res=True):
        """
        Multi-head self attention layer for AutoInt
        :param att_size: int. attention embedding size
        :param head_num: int. head num
        :param use_res: bool. use res block or not
        """
        super(InteractLayer, self).__init__()
        self.att_size = att_size
        self.head_num = head_num
        self.use_res = use_res

    def build(self, input_shape):
        self.W_Query = self.add_weight(name='query',
                                       shape=[int(input_shape[-1]), self.att_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=1))
        self.W_key = self.add_weight(name='key',
                                     shape=[int(input_shape[-1]), self.att_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=2))
        self.W_Value = self.add_weight(name='value',
                                       shape=[int(input_shape[-1]), self.att_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=3))
        if self.use_res:
            self.W_Res = self.add_weight(name='res',
                                         shape=[int(input_shape[-1]), self.att_size * self.head_num],
                                         dtype=tf.float32,
                                         initializer=TruncatedNormal(seed=4))

    def call(self, inputs, **kwargs):

        # [Batch_size, field_dim, att_size*head_num]
        query = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        # [head_num, Batch_size, field_dim, att_size]
        query = tf.stack(tf.split(query, self.head_num, axis=2), axis=0)
        keys = tf.stack(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.stack(tf.split(values, self.head_num, axis=2), axis=0)

        # [head_num, Batch_size, field_dim, field_dim]
        similarity = tf.matmul(query, keys, transpose_b=True)

        # [head_num, Batch_size, field_dim, field_dim]
        normalized_att_scores = tf.nn.softmax(similarity)

        # [head_num, Batch_size, field_dim, att_size]
        result = tf.matmul(normalized_att_scores, values)

        # [Batch_size, field_dim, field_dim*att_size]
        result = tf.squeeze(tf.concat(tf.split(result, self.head_num,), axis=-1), axis=0)

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))

        result = tf.nn.relu(result)

        return result