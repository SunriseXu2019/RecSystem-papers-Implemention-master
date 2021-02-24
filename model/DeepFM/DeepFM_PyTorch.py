import torch
import torch.nn as nn
from Layers.Layers_PyTorch import Linear, DNNLayer


class DeepFM(torch.nn.Module):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, dnn_units, dropout_rate, use_bn,
                 device):
        """
        DeepFM
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param dnn_units: list. dnn layer size for deep part
        :param dropout_rate: float.
        :param use_bn: bool.
        """
        super(DeepFM, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns

        self.linear_size = len(self.dense_feature_columns)

        self.dnn_layers = [len(dense_feature_columns) + embed_dim * len(sparse_feature_columns)] + list(dnn_units)

        self.embedding = nn.ModuleDict({
            'embed_' + str(i): torch.nn.Embedding(feat['feat_num'], embed_dim).to(device)
            for i, feat in enumerate(self.sparse_feature_columns)})

        for embed in self.embedding.values():
            nn.init.xavier_normal_(embed.weight, gain=1)

        self.linear = Linear(self.linear_size, device=device)

        self.dnn = DNNLayer(self.dnn_layers, dropout_rate=dropout_rate, use_bn=use_bn, device=device)

        self.dnn_linear = nn.Linear(self.dnn_layers[-1], 1, bias=False).to(device)
        nn.init.xavier_normal_(self.dnn_linear.weight.data, gain=1)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        sparse_embed = torch.cat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.size()[1])], dim=-1)

        linear_result = self.linear(dense_input, sparse_embed)

        fm_result = 0.5 * torch.sum(torch.sum(sparse_embed, dim=1, keepdim=True) ** 2
                                    - torch.sum(sparse_embed ** 2, dim=1, keepdim=True), dim=1, keepdim=True)

        dnn_result = self.dnn_linear(self.dnn(torch.cat([dense_input, sparse_embed], dim=-1)))

        return torch.sigmoid(linear_result + fm_result + dnn_result)


