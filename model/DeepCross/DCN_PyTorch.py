import torch
import torch.nn as nn
from Layers.Layers_PyTorch import Linear, DNNLayer, CrossLayer


class DCN(torch.nn.Module):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, cross_num, dnn_units, dropout_rate,
                 use_bn, device):
        """
        Deep & Cross
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param cross_num: int.
        :param dnn_units: list. dnn layer size for deep part
        """
        super(DCN, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns

        self.cross_num = cross_num
        self.embed_dim = embed_dim

        self.linear_size = len(self.dense_feature_columns)

        self.field_dim = len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)

        self.dnn_layers = [len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)] + list(dnn_units)

        self.embedding = {
            'embed_' + str(i): torch.nn.Embedding(feat['feat_num'], embed_dim, max_norm=0.1).to(device)
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.linear = Linear(self.linear_size, device=device)

        self.cross = CrossLayer(cross_num, self.field_dim, device)

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, device=device)

        self.dnn_linear = nn.Linear(self.field_dim + self.dnn_layers[-1], 1, bias=False).to(device)

        self.dnn_linear.weight.data.normal_(mean=0, std=0.01)
        # nn.init.xavier_normal_(self.dnn_linear, gain=1)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        sparse_embed = torch.cat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.size()[1])], dim=-1)
        # [Batch, 1]
        linear_result = self.linear(dense_input, sparse_embed)

        # [Batch, dense + sparse * embed_size]
        dense_sparse_concat = torch.cat([dense_input, sparse_embed], dim=-1)

        # [Batch, dense + sparse * embed_size]
        cross_output = self.cross(dense_sparse_concat)

        # [Batch, dnn_out_size]
        dnn_output = self.dnn(dense_sparse_concat)

        feature_cross_result = self.dnn_linear(torch.cat([cross_output, dnn_output], dim=1))

        return torch.sigmoid(linear_result + feature_cross_result)

