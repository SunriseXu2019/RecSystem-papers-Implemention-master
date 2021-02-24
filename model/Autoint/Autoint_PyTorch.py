import torch
import torch.nn as nn
from Layers.Layers_PyTorch import Linear, DNNLayer, InteractLayer


class AutoInt(torch.nn.Module):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, att_layer_size, att_size,
                 head_num, dnn_units, dropout_rate, use_bn, device):
        """
        AutoInt
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param att_layer_size: int. attention layer size
        :param att_size: int. attention size
        :param head_num: int.
        :param dnn_units: list. dnn layer size for deep part
        """
        super(AutoInt, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.linear_size = len(self.dense_feature_columns)

        self.att_layer_size = att_layer_size

        self.dnn_layers = [embed_dim * (len(sparse_feature_columns) + len(dense_feature_columns))] + list(dnn_units)

        self.sparse_embedding = nn.ModuleDict({
            'embed_' + str(i): torch.nn.Embedding(feat['feat_num'], embed_dim).to(device)
            for i, feat in enumerate(self.sparse_feature_columns)})

        self.dense_embedding = nn.ParameterList(nn.Parameter(torch.empty(embed_dim))
                                                for _ in range(len(dense_feature_columns)))
        for embed in self.dense_embedding:
            nn.init.normal_(embed, mean=0, std=0.0001)

        for embed in self.sparse_embedding.values():
            nn.init.normal_(embed.weight, mean=0, std=0.0001)

        self.linear = Linear(self.linear_size, device=device)

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, device=device)

        self.interact = nn.ModuleList([InteractLayer(input_size=embed_dim if i == 0 else att_size * head_num,
                                                     att_size=att_size,
                                                     head_num=head_num,
                                                     use_res=True)
                                       for i in range(self.att_layer_size)])

        self.dense = nn.Linear(in_features=head_num * att_size
                                           * (len(sparse_feature_columns) + len(dense_feature_columns))
                                           + self.dnn_layers[-1],
                               out_features=1,
                               bias=False).to(device)
        nn.init.xavier_normal_(self.dense.weight.data, gain=1)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        sparse_embed = torch.stack([self.sparse_embedding['embed_{}'.format(i)](sparse_input[:, i])
                                    for i in range(sparse_input.shape[1])], dim=1)

        dense_embed = torch.stack([torch.tensordot(dense_input[:, i], self.dense_embedding[i], dims=0)
                                   for i in range(dense_input.shape[1])], dim=1)

        linear_result = self.linear(dense_input, torch.flatten(sparse_embed, start_dim=1))

        dnn_output = self.dnn(torch.cat([torch.flatten(dense_embed, start_dim=1),
                                         torch.flatten(sparse_embed, start_dim=1)], dim=1))

        interact_input = torch.cat([dense_embed, sparse_embed], dim=1)

        for i in range(self.att_layer_size):
            interact_input = self.interact[i](interact_input)

        interact_output = torch.flatten(interact_input, start_dim=1)

        autoint_result = self.dense(torch.cat([dnn_output, interact_output], dim=1))

        return torch.sigmoid(linear_result + autoint_result)


