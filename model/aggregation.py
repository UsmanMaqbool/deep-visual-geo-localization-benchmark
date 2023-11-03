
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

import model.functional as LF
import model.normalization as normalization
from os.path import join, exists, isfile, realpath, dirname

# # graphsage
import torch.nn.init as init

# # Semantic Segmentation
# 
# # import espnet as net
# from .espnet import *

class MAC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.mac(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

class SPoC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.spoc(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps
    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper 
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale 
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """
    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
        self.l2 = normalization.L2Norm()
    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.l2(self.ln2(out))
        return out


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        # initcache = join(args.dataPath, args.arch + '_pitts_' + str(args.num_clusters) +'_desc_cen.hdf5')
        initcache = "/home/leo/usman_ws/datasets/2015netVLAD/official/vgg16_pitts_64_desc_cen.hdf5"
        
        if not exists(initcache):
            raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

        with h5py.File(initcache, mode='r') as h5: 
            self.init_params(h5.get("centroids")[...], h5.get("descriptors")[...])


        self = self.to(args.device)


class CRNModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Downsample pooling
        self.downsample_pool = nn.AvgPool2d(kernel_size=3, stride=(2, 2),
                                            padding=0, ceil_mode=True)
        
        # Multiscale Context Filters
        self.filter_3_3 = nn.Conv2d(in_channels=dim, out_channels=32,
                                    kernel_size=(3, 3), padding=1)
        self.filter_5_5 = nn.Conv2d(in_channels=dim, out_channels=32,
                                    kernel_size=(5, 5), padding=2)
        self.filter_7_7 = nn.Conv2d(in_channels=dim, out_channels=20,
                                    kernel_size=(7, 7), padding=3)
        
        # Accumulation weight
        self.acc_w = nn.Conv2d(in_channels=84, out_channels=1, kernel_size=(1, 1))
        # Upsampling
        self.upsample = F.interpolate
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize Context Filters
        torch.nn.init.xavier_normal_(self.filter_3_3.weight)
        torch.nn.init.constant_(self.filter_3_3.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_5_5.weight)
        torch.nn.init.constant_(self.filter_5_5.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_7_7.weight)
        torch.nn.init.constant_(self.filter_7_7.bias, 0.0)
        
        torch.nn.init.constant_(self.acc_w.weight, 1.0)
        torch.nn.init.constant_(self.acc_w.bias, 0.0)
        self.acc_w.weight.requires_grad = False
        self.acc_w.bias.requires_grad = False
    
    def forward(self, x):
        # Contextual Reweighting Network
        x_crn = self.downsample_pool(x)
        
        # Compute multiscale context filters g_n
        g_3 = self.filter_3_3(x_crn)
        g_5 = self.filter_5_5(x_crn)
        g_7 = self.filter_7_7(x_crn)
        g = torch.cat((g_3, g_5, g_7), dim=1)
        g = F.relu(g)
        
        w = F.relu(self.acc_w(g))  # Accumulation weight
        mask = self.upsample(w, scale_factor=2, mode='bilinear')  # Reweighting Mask
        
        return mask


class CRN(NetVLAD):
    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        super().__init__(clusters_num, dim, normalize_input)
        self.crn = CRNModule(dim)
    
    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        
        mask = self.crn(x)
        
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # Weight soft_assign using CRN's mask
        soft_assign = soft_assign * mask.view(N, 1, H * W)
        
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[D:D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D + 1, :].unsqueeze(2)
            vlad[:, D:D + 1, :] = residual.sum(dim=-1)
        
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad


#### graphsage     
class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """Aggregate node neighbors

        Args:
            input_dim: the dimension of the input feature
            output_dim: the dimension of the output feature
            use_bias: whether to use bias (default: {False})
            aggr_method: neighbor aggregation method (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
       # print(neighbor_feature.shape)
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = torch.amax(neighbor_feature, 1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        # print(aggr_neighbor.shape,self.weight.shape)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
    
#F.pre PReLU
# prelu
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.gelu,
                 aggr_neighbor_method="sum",
                 aggr_hidden_method="concat"):
        """SageGCN layer definition
        # firstworking with mean and concat
        Args:
            input_dim: the dimension of the input feature
            hidden_dim: dimension of hidden layer features,
                When aggr_hidden_method=sum, the output dimension is hidden_dim
                When aggr_hidden_method=concat, the output dimension is hidden_dim*2
            activation: activation function
            aggr_neighbor_method: neighbor feature aggregation method, ["mean", "sum", "max"]
            aggr_hidden_method: update method of node features, ["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        
        # print('src_node_features', neighbor_node_features.shape, src_node_features.shape, self.weight.shape)
        self_hidden = torch.matmul(src_node_features, self.weight)
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim #1433
        self.hidden_dim = hidden_dim #[128, 7]
        self.num_neighbors_list = num_neighbors_list #[10, 10]
        self.num_layers = len(num_neighbors_list)  #2
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0])) # (1433, 128)
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1])) #128, 7
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
        


    def forward(self, node_features_list):
        hidden = node_features_list
        # code.interact(local=locals())
        subfeat_size = int(hidden[0].shape[1]/self.input_dim)
        gcndim = int(self.input_dim) 
        
        # print('subfeat_size ', subfeat_size)
        # print('  l  ', ' hop  ', '  src_node_features  ', '  neighbor_node_features  ', '  h  ', '    ')

        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # print('neighbor_node_features ', hidden[hop + 1].shape  ,' / ',  src_node_num, self.num_neighbors_list[hop], '-1')
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                # print(l,' ', hop  ,'  ',  src_node_features.shape  ,'  ' , neighbor_node_features.shape)
                
                # splitting the i/p
                #h = gcn(src_node_features, neighbor_node_features)
                for j in range(subfeat_size): 
                    h_x = gcn(src_node_features[:,j*gcndim:j*gcndim+gcndim], neighbor_node_features[:,:,j*gcndim:j*gcndim+gcndim])
                    # neighborsFeat = []
                    if (j==0):
                        h = h_x;
                    else:
                        h = torch.concat([h, h_x],1) 
                        
                # print("hop", hop,'  ',  h.shape)
                next_hidden.append(h)
            hidden = next_hidden
        # print("hidden", ' ',  hidden[0].shape)    
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )