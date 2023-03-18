import torch
import  math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dim):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.att_A, self.att_A2, self.att_mlp = 0, 0, 0
        self.hidden = dim
        self.W_k0, self.W_k1, self.W_k2 = Parameter(
                torch.FloatTensor(out_features,self.hidden)), Parameter(
                torch.FloatTensor(out_features,self.hidden)), Parameter(torch.FloatTensor(out_features,self.hidden))
        self.weight_A, self.weight_A2, self.weight_mlp = Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
        self.att_vec_A, self.att_vec_A2, self.att_vec_mlp = Parameter(
                torch.FloatTensor(out_features, self.hidden)), Parameter(torch.FloatTensor(out_features, self.hidden)), Parameter(
                torch.FloatTensor(out_features, self.hidden))
        self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))
        std_k = 1./math.sqrt(self.W_k0.size(1))
        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))

        self.W_k0.data.uniform_(-std_k, std_k)
        self.W_k1.data.uniform_(-std_k, std_k)
        self.W_k2.data.uniform_(-std_k, std_k)

        self.weight_A.data.uniform_(-stdv, stdv)
        self.weight_A2.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.att_vec_A.data.uniform_(-std_att, std_att)
        self.att_vec_A2.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_A, output_A2, output_mlp):  #
        T = 3
        k0 = torch.mean(torch.mm(output_A,self.W_k0), dim=0, keepdim=True)
        k1 = torch.mean(torch.mm(output_A2, self.W_k1), dim=0, keepdim=True)
        k2 = torch.mean(torch.mm(output_mlp, self.W_k2), dim=0, keepdim=True)

        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat(
            [torch.mm(torch.mm((output_A), self.att_vec_A),k0.T), torch.mm(torch.mm((output_A2), self.att_vec_A2),k1.T),
            torch.mm(torch.mm((output_mlp), self.att_vec_mlp),k2.T)], 1)), self.att_vec) / T, 1)  #


        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, inputx , adj_A, adj_A2):
        output_A = F.relu(torch.spmm(adj_A, (torch.mm(inputx, self.weight_A))))
        output_A2 = F.relu(torch.spmm(adj_A2, (torch.mm(inputx, self.weight_A2))))
        output_mlp = F.relu(torch.mm(inputx, self.weight_mlp))

        self.att_A, self.att_A2, self.att_mlp = self.attention((output_A), (output_A2), (output_mlp))  #
        return 3 * (self.att_A * output_A + self.att_A2 * output_A2 + self.att_mlp * output_mlp)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SHGCN(nn.Module):
    def __init__(self, dataset,args):
        super(SHGCN, self).__init__()
        self.gcns = nn.ModuleList()
        self.dropout = args.dropout

        self.gcns.append(GraphConvolution(dataset.num_features, args.hidden, args.dim))
        self.gcns.append(GraphConvolution(args.hidden, dataset.num_classes, args.dim))


    def forward(self, data, adj_A, adj_A2):
        x = data.x
        x = F.dropout(x, self.dropout, training=self.training)
        fea = (self.gcns[0](x, adj_A, adj_A2))
        fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj_A, adj_A2)

        return F.log_softmax(fea, dim=1)



