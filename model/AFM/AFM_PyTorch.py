import torch
import torch.nn.functional as functional
from Layers.Layers_PyTorch import Linear


class AFM(torch.nn.Module):

    def __init__(self, dense_feature_columns, sparse_feature_columns, embed_dim, att_size, dropout_rate, device):
        """
        :param dense_feature_columns: list. dense features information:{'name': feat}
        :param sparse_feature_columns: list. sparse features information:{'name': feat, 'feat_num': feat_num}
        :param embed_dim: int. embedding size for sparse features
        :param att_size: int. attention size
        :param dropout_rate: float. dropout rate
        """
        super(AFM, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns

        self.linear_size = len(self.dense_feature_columns)
        self.att_size = att_size
        self.embed_dim = embed_dim

        self.field_dim = [feat['feat_num'] for feat in self.sparse_feature_columns]

        self.att_W = torch.nn.Parameter(torch.empty(self.embed_dim, self.att_size))

        self.att_b = torch.nn.Parameter(torch.empty(self.att_size))

        self.att_h = torch.nn.Parameter(torch.empty(self.att_size, 1))

        self.att_p = torch.nn.Parameter(torch.empty(self.embed_dim, 1))

        self.embedding = {
            'embed_' + str(i): torch.nn.Embedding(feat['feat_num'], embed_dim).to(device)
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.linear = Linear(self.linear_size, device=device)

        for weight in [self.att_W, self.att_h, self.att_p]:
            torch.nn.init.xavier_normal_(weight, gain=1)

        for bias in [self.att_b]:
            torch.nn.init.zeros_(bias)

        for _, embed in self.embedding.items():
            torch.nn.init.normal_(embed.weight, mean=0, std=0.001)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, dense_input, sparse_input):

        sparse_embed_list = [self.embedding['embed_{}'.format(i)](sparse_input[:, i])
                             for i in range(sparse_input.size()[1])]

        # [Batch, field_num*embed_dim]
        sparse_concat = torch.cat(sparse_embed_list, dim=1)

        # [Batch, field_num, embed_dim]
        sparse_stack = torch.stack(sparse_embed_list, dim=1)

        # [Batch, 1]
        linear_result = self.linear(dense_input, sparse_concat)

        element_wise_product_list = []
        for i in range(len(self.field_dim) - 1):
            for j in range(i + 1, len(self.field_dim)):
                element_wise_product_list.append(sparse_stack[:, i, :] * sparse_stack[:, j, :])

        # [Batch, field_num*(field_num - 1)/2, embed_dim]
        element_wise_product = torch.stack(element_wise_product_list, dim=1)

        # [Batch, field_num*(field_num - 1)/2, att_size]
        att_wx_b = functional.relu(torch.tensordot(element_wise_product, self.att_W, dims=([-1], [0])) + self.att_b)

        # [Batch, field_num*(field_num - 1)/2, 1]
        normalized_att_score = functional.softmax(torch.tensordot(att_wx_b, self.att_h, dims=([-1], [0])), dim=1)

        # [Batch, embed_dim]
        att_output = torch.sum(normalized_att_score * element_wise_product, dim=1)

        att_output = self.dropout(att_output)

        # [Batch, 1]
        afm_result = torch.tensordot(att_output, self.att_p, dims=([-1], [0]))

        return torch.sigmoid(afm_result + linear_result)


