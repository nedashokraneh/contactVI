import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class GraphConv(nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 feature = True,
                 act = None):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature = feature
        self.weight = nn.Parameter(torch.DoubleTensor(input_dim, output_dim))
        self.init_parameters()
        self.act = act

    def init_parameters(self):
        # from https://github.com/DuYooho/DGVAE_pytorch/blob/main/layers.py
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weight, -init_range, init_range)

    def forward(self, x, adj):
        if self.feature:
            x = torch.matmul(x, self.weight)
        else:
            x = self.weight
        x = torch.matmul(adj, x)
        if self.act is not None:
            x = self.act(x)
        return (x)


class InnerProductDecoder(nn.Module):

    def __init__(self, act = None):
        
        super(InnerProductDecoder, self).__init__()
        self.act = act 
        
    def forward(self, inputs):

        x = torch.matmul(inputs, inputs.transpose(1,0))
        if self.act is not None:
            x = self.act(x)
        return x

class AltInnerProductDecoder(nn.Module):

    def __init__(self,
                 n_nodes,
                 n_latent,
                 #n_hidden,
                 distance_effect = True,
                 norm = True,
                 device = "cpu"
                ):
        
        super().__init__()

        self.n_nodes = n_nodes
        self.n_latent = n_latent
        #self.n_hidden = n_hidden
        self.distance_effect = distance_effect
        self.input_dim = self.n_latent 
        self.ut_idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        self.device = device
        if self.distance_effect:
            self.input_dim += 1
            col_idx = torch.arange(0, self.n_nodes).expand(self.n_nodes, -1)
            self.dist_per_pair = abs(col_idx - torch.t(col_idx))[self.ut_idx[0, :], self.ut_idx[1, :]].to(self.device)
        self.n_hidden = int(self.input_dim / 2)
        self.decoder = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(self.input_dim, self.n_hidden)),
            ('act1', nn.ReLU()),
            ('dense2', nn.Linear(self.n_hidden, 1))]))
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(self.input_dim)

    def forward(self, z):
        
        z_combined = torch.cat((z.unsqueeze(1).repeat(1, self.n_nodes, 1),
                                z.unsqueeze(0).repeat_interleave(self.n_nodes, dim = 0)),
                               dim = -1)
        z_combined = z_combined[self.ut_idx[0, :], self.ut_idx[1, :], :]
        z_product = torch.mul(z_combined[:, :self.n_latent], z_combined[:, self.n_latent:])
        if self.distance_effect:
            z_product = torch.concat([z_product, self.dist_per_pair.reshape(-1, 1)], axis = 1)
        if self.norm:
            z_product = self.layer_norm(z_product)
        output = self.decoder(z_product)
        unnorm_adj = torch.zeros((self.n_nodes, self.n_nodes)).double().to(self.device)
        unnorm_adj[self.ut_idx[0, :], self.ut_idx[1, :]] = output.squeeze(1)
        return unnorm_adj
        
            
class JointDecoder(nn.Module):
    
    def __init__(self,
                 n_nodes,
                 n_latent,
                 #n_hidden,
                 distance_effect = True,
                 norm = True, 
                 device = "cpu"
                ):

        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_latent = n_latent
        #self.n_hidden = n_hidden
        self.distance_effect = distance_effect
        self.input_dim = 2 * self.n_latent 
        self.ut_idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        self.device = device
        if self.distance_effect:
            self.input_dim += 1
            col_idx = torch.arange(0, self.n_nodes).expand(self.n_nodes, -1)
            self.dist_per_pair = abs(col_idx - torch.t(col_idx))[self.ut_idx[0, :], self.ut_idx[1, :]].to(self.device)
        self.n_hidden = int(self.input_dim / 2)
        self.decoder = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(self.input_dim, self.n_hidden)),
            ('act1', nn.ReLU()),
            ('dense2', nn.Linear(self.n_hidden, 1))]))
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(self.input_dim)

    def forward(self, z):
        
        z_combined = torch.cat((z.unsqueeze(1).repeat(1, self.n_nodes, 1),
                                z.unsqueeze(0).repeat_interleave(self.n_nodes, dim = 0)),
                               dim = -1)
        z_combined = z_combined[self.ut_idx[0, :], self.ut_idx[1, :], :]
        if self.distance_effect:
            z_combined = torch.concat([z_combined, self.dist_per_pair.reshape(-1, 1)], axis = 1)
        if self.norm:
            z_combined = self.layer_norm(z_combined)
        output = self.decoder(z_combined)
        unnorm_adj = torch.zeros((self.n_nodes, self.n_nodes)).double().to(self.device)
        unnorm_adj[self.ut_idx[0, :], self.ut_idx[1, :]] = output.squeeze(1)
        return unnorm_adj

class CombinedDecoder(nn.Module):
    
    def __init__(self,
                 n_nodes,
                 n_latent,
                 #n_hidden,
                 distance_effect = True,
                 norm = True,
                 device = "cpu"
                ):
        
        super().__init__()

        self.n_nodes = n_nodes
        self.n_latent = n_latent
        #self.n_hidden = n_hidden
        self.distance_effect = distance_effect
        self.input_dim = 3 * self.n_latent 
        self.ut_idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        self.device = device
        if self.distance_effect:
            self.input_dim += 1
            col_idx = torch.arange(0, self.n_nodes).expand(self.n_nodes, -1)
            self.dist_per_pair = abs(col_idx - torch.t(col_idx))[self.ut_idx[0, :], self.ut_idx[1, :]].to(self.device)
        self.n_hidden = int(self.input_dim / 2)
        self.decoder = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(self.input_dim, self.n_hidden)),
            ('act1', nn.ReLU()),
            ('dense2', nn.Linear(self.n_hidden, 1))]))
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(self.input_dim)

    def forward(self, z):
        
        z_combined = torch.cat((z.unsqueeze(1).repeat(1, self.n_nodes, 1),
                                z.unsqueeze(0).repeat_interleave(self.n_nodes, dim = 0)),
                               dim = -1)
        z_combined = z_combined[self.ut_idx[0, :], self.ut_idx[1, :], :]
        z_product = torch.mul(z_combined[:, :self.n_latent], z_combined[:, self.n_latent:])
        z_combined = torch.concat([z_combined, z_product], axis = 1)
        if self.distance_effect:
            z_combined = torch.concat([z_combined, self.dist_per_pair.reshape(-1, 1)], axis = 1)
        if self.norm:
            z_combined = self.layer_norm(z_combined)
        output = self.decoder(z_combined)
        unnorm_adj = torch.zeros((self.n_nodes, self.n_nodes)).double().to(self.device)
        unnorm_adj[self.ut_idx[0, :], self.ut_idx[1, :]] = output.squeeze(1)
        return unnorm_adj
        