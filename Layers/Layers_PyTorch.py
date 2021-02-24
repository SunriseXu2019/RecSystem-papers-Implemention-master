import torch
import torch.nn as nn
import torch.nn.functional as functional


class Linear(torch.nn.Module):

    def __init__(self, linear_size, device):
        """
        :param linear_size: int. dense features dim
        :param device:
        """
        super(Linear, self).__init__()
        self.Linear = torch.nn.Linear(linear_size, 1)
        nn.init.zeros_(self.Linear.bias.data)
        nn.init.xavier_normal_(self.Linear.weight.data, gain=1)
        self.to(device)

    def forward(self, dense_input, sparse_embed):

        linear_sparse = torch.sum(sparse_embed, dim=1, keepdim=True)

        linear_dense = self.Linear(dense_input)

        return linear_dense + linear_sparse


class DNNLayer(torch.nn.Module):
    def __init__(self, dnn_layers, dropout_rate, use_bn, device):
        """
        :param dnn_layers: list. dnn layer size
        :param dropout_rate: float. dropout rate
        :param use_bn: bool.
        """
        super(DNNLayer, self).__init__()
        self.dnn_layers = dnn_layers
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        self.linear = nn.ModuleList(nn.Linear(self.dnn_layers[i], self.dnn_layers[i + 1])
                                    for i in range(len(self.dnn_layers) - 1))

        if self.use_bn:
            self.bn = nn.ModuleList(nn.BatchNorm1d(self.dnn_layers[i + 1]) for i in range(len(self.dnn_layers) - 1))

        for name, weight in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(weight, gain=1)
        self.to(device)

    def forward(self, inputs):

        # [B, dense_dim + embed_dim * sparse_dim]
        dnn_input = inputs
        for i in range(len(self.linear)):

            # nn.Linear中自带bias
            x = self.linear[i](dnn_input)
            if self.use_bn:
                x = self.bn[i](x)
            x = nn.ReLU(inplace=True)(x)
            x = self.dropout(x)
            dnn_input = x
        return dnn_input


class CrossLayer(nn.Module):
    def __init__(self, cross_num, field_dim, device):
        super(CrossLayer, self).__init__()
        self.cross_num = cross_num
        self.field_dim = field_dim
        self.weight = nn.ParameterList([nn.Parameter(torch.empty(self.field_dim, 1)) for _ in range(self.cross_num)])

        self.bias = nn.ParameterList([nn.Parameter(torch.empty(self.field_dim, )) for _ in range(self.cross_num)])

        for weight in self.weight:
            nn.init.xavier_normal_(weight, gain=1)
        for bias in self.bias:
            nn.init.zeros_(bias)

        self.to(device)

    def forward(self, x):
        x_0 = x
        x_l = x_0
        for i in range(self.cross_num):
            x_l += torch.mul(x_0, torch.tensordot(x_l, self.weight[i], dims=([1], [0]))) + self.bias[i]
        return x_l


class CIN(nn.Module):
    def __init__(self, field_dim, cin_layers, direct, device):
        super(CIN, self).__init__()
        self.cin_layers = cin_layers
        self.direct = direct
        if self.direct:
            self.field_dim = [field_dim] + list(layer for layer in self.cin_layers)
        else:
            self.field_dim = [field_dim] + list(int(layer / 2) for layer in self.cin_layers)

        self.output_size = sum(self.field_dim[1:])
        self.conv_layers = nn.ModuleList(nn.Conv1d(self.field_dim[i] * self.field_dim[0], layer, 1)
                                         for i, layer in enumerate(self.cin_layers))

        self.cin_linear = nn.Linear(self.output_size, 1, bias=False).to(device)

        nn.init.xavier_normal_(self.cin_linear.weight.data, gain=1)
        for conv in self.conv_layers:
            nn.init.xavier_normal_(conv.weight, gain=1)
        self.to(device)

    def forward(self, inputs):
        """
        :param inputs: [Batch, sparse_dim, embed_size]
        :return:
        """
        # batch_size = inputs.shape[0]
        # dim = inputs.shape[-1]
        cin_inputs = [inputs]
        cin_result = []

        for i, layer_size in enumerate(self.cin_layers):

            # [Batch, Hk, sparse_dim, embed_size]
            x = torch.einsum('ihk, imk -> ihmk', cin_inputs[-1], cin_inputs[0])

            # [Batch, Hk*sparse_dim, embed_size]
            x = x.reshape(inputs.shape[0], cin_inputs[-1].shape[1] * cin_inputs[0].shape[1], inputs.shape[-1])

            # [Batch, Hk+1, embed_size]
            x = self.conv_layers[i](x)

            curr_out = nn.ReLU()(x)

            if self.direct:
                cin_inputs.append(curr_out)
                cin_result.append(curr_out)
            else:
                if i <= len(self.cin_layers) - 1:
                    a, b = torch.split(curr_out, [int(layer_size / 2)] * 2, 1)
                    cin_result.append(a)
                    cin_inputs.append(b)
                else:
                    cin_result.append(curr_out)

        cin_result = torch.sum(torch.cat(cin_result, dim=1), dim=-1)

        return self.cin_linear(cin_result)


class InteractLayer(torch.nn.Module):
    def __init__(self, input_size, att_size, head_num, use_res):
        super(InteractLayer, self).__init__()
        self.input_size = input_size
        self.att_size = att_size
        self.head_num = head_num
        self.use_res = use_res

        self.W_Query = torch.nn.Parameter(torch.empty(self.input_size, self.att_size * self.head_num))
        self.W_Key = torch.nn.Parameter(torch.empty(self.input_size, self.att_size * self.head_num))
        self.W_Values = torch.nn.Parameter(torch.empty(self.input_size, self.att_size * self.head_num))

        if self.use_res:
            self.W_Res = torch.nn.Parameter(torch.empty(self.input_size, self.att_size * self.head_num))
            nn.init.xavier_normal_(self.W_Res, gain=1)

        for weight in [self.W_Query, self.W_Key, self.W_Values]:
            nn.init.xavier_normal_(weight, gain=1)

        # if self.use_res:
        #     self.W_res = torch.nn.Parameter(torch.empty(self.input_size, self.att_size * self.head_num))
        #     for weight in self.W_res:
        #         nn.init.xavier_normal_(weight, gain=1)

    def forward(self, inputs):

        # [Batch_size, field_dim, att_size*head_num]
        query = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Values, dims=([-1], [0]))

        # [head_num, Batch_size, field_dim, att_size]
        query = torch.stack(torch.split(query, self.att_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, self.att_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, self.att_size, dim=2), dim=0)

        # [head_num, Batch_size, field_dim, field_dim]
        similarity = torch.matmul(query, keys.transpose(2, -1))

        # [head_num, Batch_size, field_dim, field_dim]
        normalized_att_scores = functional.softmax(similarity)

        # [head_num, Batch_size, field_dim, att_size]
        result = torch.matmul(normalized_att_scores, values)

        # [Batch_size, field_dim, field_dim*att_size]
        result = torch.squeeze(torch.cat(torch.split(result, 1, dim=0), dim=-1), dim=0)

        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))

        result = nn.ReLU()(result)

        return result