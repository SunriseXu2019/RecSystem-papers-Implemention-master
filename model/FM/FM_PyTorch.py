import torch
import torch.nn as nn
from Layers.Layers_PyTorch import Linear


class FMLayer(torch.nn.Module):

    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, device):
        super(FMLayer, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns

        self.linear_size = len(self.dense_feature_columns)
        self.embed_dim = embed_dim

        self.embedding = nn.ModuleDict({
            'embed_' + str(i): torch.nn.Embedding(feat['feat_num'], embed_dim).to(device)
            for i, feat in enumerate(self.sparse_feature_columns)})

        for _, embedding in self.embedding.items():
            torch.nn.init.xavier_normal_(embedding.weight, gain=1)

        self.linear = Linear(self.linear_size, device)
        self.to(device)

    def forward(self, dense_input, sparse_input):

        # [Batch, field * embed_size]
        sparse_embed = torch.cat([self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                                  for i in range(sparse_input.size()[1])], dim=1)

        linear_result = self.linear(dense_input, sparse_embed)

        feature_cross_result = 0.5 * torch.sum(torch.sum(sparse_embed, dim=1, keepdim=True) ** 2
                                               - torch.sum(sparse_embed ** 2, dim=1, keepdim=True), dim=1, keepdim=True)

        output = torch.sigmoid(linear_result + feature_cross_result)

        return output


