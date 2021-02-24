import numpy as np
import torch


class Linear(torch.nn.Module):
    def __init__(self, linear_size, field_dim, device):
        super(Linear, self).__init__()
        self.linear_size = linear_size
        self.field_dim = field_dim
        self.weight_dense = torch.nn.Linear(self.linear_size, 1)
        self.weight_sparse = torch.nn.Embedding(sum(self.field_dim), 1)

        # init weight
        self.weight_dense.bias.data.zero_()
        self.weight_dense.weight.data.normal_(mean=0, std=0.001)
        torch.nn.init.normal_(self.weight_sparse.weight, mean=0, std=0.001)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        linear_sparse = torch.sum(self.weight_sparse(sparse_input), dim=1)

        linear_dense = self.weight_dense(dense_input)

        return linear_dense + linear_sparse


class FFMLayer(torch.nn.Module):
    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, device):
        super(FFMLayer, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns

        self.linear_size = len(self.dense_feature_columns)
        self.field_dim = [feat['feat_num'] for feat in self.sparse_feature_columns]

        self.embed_dim = embed_dim
        self.embedding = {
            'field_' + str(i): torch.nn.Embedding(sum(self.field_dim), embed_dim, max_norm=0.01).to(device)
            for i in range(len(self.field_dim))
        }
        self.offset = torch.tensor(np.array((0, *np.cumsum(self.field_dim)[:-1]),
                                            dtype=np.long)).unsqueeze(0).to(device)

        self.linear = Linear(self.linear_size, self.field_dim, device)
        for _, embed in self.embedding.items():
            torch.nn.init.xavier_normal_(embed.weight, gain=1)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        # 构建one-hot的索引形式，self.offset保存sparse info中各个特征域的维度和
        sparse_input = sparse_input + self.offset

        linear_result = self.linear(dense_input, sparse_input)

        sparse_embed = [self.embedding['field_{}'.format(i)](sparse_input)
                        for i in range(len(self.field_dim))]
        feature_cross = []
        for i in range(len(self.field_dim) - 1):
            for j in range(i + 1, len(self.field_dim)):
                feature_cross.append(torch.sum(sparse_embed[j][:, i] * sparse_embed[i][:, j], dim=1, keepdim=True))

        feature_cross_result = torch.sum(torch.stack(feature_cross, dim=1), dim=1)

        return torch.sigmoid(feature_cross_result + linear_result)





