from typing import Literal
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal, Distribution
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from modules import GraphConv, InnerProductDecoder, AltInnerProductDecoder, JointDecoder, CombinedDecoder


class GraphEncoder(nn.Module):

    def __init__(self, 
                 variational,
                 n_good_nodes,
                 n_hidden,
                 n_latent,
                 activation,
                 var_eps,
                 device,
                 node_features = None,
                 self_adj_input = False
                ):
        
        super().__init__()
        
       
        self.variational = variational
        self.var_eps = var_eps
        self.n_good_nodes = n_good_nodes
        self.device = device
        l1_feature = (node_features is not None) or self_adj_input

        #self.encoder = GraphConv(n_good_nodes, n_hidden, feature = False, act = activation)
        self.encoder = GraphConv(n_good_nodes, n_hidden, feature = l1_feature, act = activation)
        self.mean_encoder = GraphConv(n_hidden, n_latent, feature = True, act = None)
        if self.variational:
            self.var_encoder = GraphConv(n_hidden, n_latent, feature = True, act = None)

        if node_features is None:
            self.node_features = torch.eye(self.n_good_nodes).double().to(self.device)
        else:
            self.node_features = node_features.double().to(self.device)
        self.self_adj_input = self_adj_input

    def forward(self,
                prop_adj = None,
                adj = None
               ):
        if self.self_adj_input:
            X = adj
        else:
            X = self.node_features
        z = self.encoder(X, prop_adj)
        z_mean = self.mean_encoder(z, prop_adj)
        if self.variational:
            z_var = self.var_encoder(z, prop_adj)
            z_var = torch.exp(z_var) + self.var_eps
            return z_mean, z_var

        return z_mean, None


class GraphDecoder(nn.Module):
    
    def __init__(self,
                 decoder_type,
                 n_nodes, 
                 n_latent,
                 distance_effect = False,
                 norm = True,
                 device = "cpu"
                ):
        
        super().__init__()
        self.decoder_type = decoder_type
        self.device = device
        if self.decoder_type == "IP":
            self.decoder = InnerProductDecoder()
        elif self.decoder_type == "alt_IP":
            self.decoder = AltInnerProductDecoder(n_nodes,
                                                  n_latent,
                                                  distance_effect,
                                                  norm,
                                                  self.device
                                                 )
        elif self.decoder_type == "joint":
            
            self.decoder = JointDecoder(n_nodes,
                                        n_latent, 
                                        distance_effect,
                                        norm,
                                        self.device
                                       )
            
        elif self.decoder_type == "combined":
            
            self.decoder = CombinedDecoder(n_nodes,
                                        n_latent, 
                                        distance_effect,
                                        norm,
                                        self.device
                                       )

        else:
            raise ValueError("Decoder type can be IP, alt_IP, joint, or combined.")

    def forward(self, z):
        unnorm_adj = self.decoder(z)
        return unnorm_adj



class Normalizer():

    def __init__(self, n_nodes, good_bins, scale_type, device = "cpu"):

        self.n_nodes = n_nodes
        self.good_bins = good_bins
        good_bp_adj = np.zeros((self.n_nodes, self.n_nodes))
        good_bp_adj[np.ix_(self.good_bins, self.good_bins)] = 1
        self.good_bp_adj = torch.tensor(good_bp_adj).bool()
        self.scale_type = scale_type
        if not self.scale_type in ["band", "pool", "whole"]:
            raise ValueError("scale type should be 'band', 'pool', or 'whole'.")
        self.device = device
        self.scale_act = nn.Softmax()
        if scale_type == "pool":
            self.pool_idx = []
            curr_pool_size = 1
            curr_start_band = 1
            while curr_start_band <= (self.n_nodes - 1):
                self.pool_idx.append(np.arange(curr_start_band, 
                                               np.min([curr_start_band + curr_pool_size, self.n_nodes]), 
                                               dtype = "int"))
                curr_start_band += curr_pool_size
                curr_pool_size += 1
        
    def normalize(self, unnorm_adj, input_adj):

        if self.scale_type == "band":
            scaled_adj = torch.zeros(unnorm_adj.shape).to(self.device)
            for d in np.arange(1, self.n_nodes):
                diag_vec = torch.diag(unnorm_adj, d)
                mask_vec = torch.diag(self.good_bp_adj, d)
                diag_sum = torch.diag(input_adj, d).sum().item()
                diag_vec_ = self.scale_act(diag_vec[mask_vec]) * diag_sum
                diag_vec[:] = 0
                diag_vec[mask_vec] = diag_vec_
                scaled_adj = torch.diagonal_scatter(scaled_adj, diag_vec, d)
        
            
        elif self.scale_type == "pool":
            scaled_adj = torch.zeros(unnorm_adj.shape).to(self.device)
            for p in range(len(self.pool_idx)):
                pool_sum = np.sum([torch.diag(input_adj, d).sum().item() for d in self.pool_idx[p]])
                pool_vec = torch.concat([torch.diag(unnorm_adj, d) for d in self.pool_idx[p]])
                mask_vec = torch.concat([torch.diag(self.good_bp_adj, d) for d in self.pool_idx[p]])
                pool_vec_ = self.scale_act(pool_vec[mask_vec]) * pool_sum
                pool_vec[:] = 0
                pool_vec[mask_vec] = pool_vec_
                
                diag_base = 0
                for d in self.pool_idx[p]:
                    diag_length = self.n_nodes - d
                    diag_vec = pool_vec[diag_base : (diag_base + diag_length)]
                    scaled_adj = torch.diagonal_scatter(scaled_adj, diag_vec, d)
                    diag_base += diag_length
    
        else:
            ut_idx = torch.triu(torch.ones(self.n_nodes, self.n_nodes), diagonal = 1) == 1
            ut_vec = unnorm_adj[ut_idx]
            mask_vec = self.good_bp_adj[ut_idx]
            whole_sum = input_adj[ut_idx].sum()
            ut_vec_ = self.scale_act(ut_vec[mask_vec]) * whole_sum
            ut_vec[:] = 0
            ut_vec[mask_vec] = ut_vec_
            scaled_adj = torch.zeros(self.n_nodes, self.n_nodes).double().to(self.device)
            scaled_adj[ut_idx] = ut_vec
            
        return scaled_adj

class GVAE(nn.Module):

    def __init__(self,
                 n_nodes: int,
                 n_hidden: int = 32,
                 n_latent: int = 16,
                 variational: bool = True,
                 likelihood: Literal["poisson", "nb", "zinb"] = "poisson",
                 dispersion: Literal["common", "distance", "bin_pair"] = "distance",
                 good_bins = None,
                 decoder_type: Literal["IP", "alt_IP", "joint", "combined"] = "IP",
                 GCN1_act = None,
                 scale_type: Literal["whole", "pool", "band"] = "whole",
                 distance_effect = True,
                 var_eps: float = 1e-4,
                 device = "cpu",
                 node_features = None,
                 self_adj_input = False,
                 norm = True
                ):

        super(GVAE, self).__init__()

        self.n_nodes = n_nodes
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.variational = variational 
        self.likelihood = likelihood
        self.good_bins = good_bins 
        if self.good_bins is not None:
            self.n_good_nodes = self.good_bins.shape[0]
            self.sub_row = self.good_bins.unsqueeze(1)
            self.sub_col = self.good_bins.repeat(self.good_bins.size(0), 1)
        else:
            self.n_good_nodes = self.n_nodes

        self.decoder_type = decoder_type
        self.scale_type = scale_type
        self.device = device
        self.norm = norm
        
        self.encoder = GraphEncoder(self.variational,
                                    self.n_good_nodes,
                                    self.n_hidden,
                                    self.n_latent,
                                    GCN1_act,
                                    var_eps,
                                    self.device,
                                    node_features,
                                    self_adj_input
                                   )
        
        self.distance_effect = distance_effect
        self.decoder = GraphDecoder(self.decoder_type,
                                    self.n_nodes,
                                    self.n_latent,
                                    self.distance_effect,
                                    self.norm, 
                                    self.device
                                   )

        self.normalizer = Normalizer(self.n_nodes, 
                                     self.good_bins, 
                                     self.scale_type, 
                                     self.device)
        
        if self.likelihood in ["nb", "zinb"]:
            self.dispersion_type = dispersion
            if dispersion == "common":
                self.dispersion = torch.nn.Parameter(torch.ones(1))
            elif dispersion == "distance":
                self.dispersion = torch.nn.Parameter(torch.ones(self.n_nodes))
            elif dispersion == "bin_pair":
                self.dispersion = torch.nn.Parameter(torch.ones(self.n_bin_pairs))
            else:
                raise ValueError("dispersion should be 'common', 'distance', or 'bin_pair'.")
                
        if self.likelihood == "zinb":
            self.dropout_func = nn.Linear(2, 1)
            self.dropout_act = nn.Sigmoid()
            
        col_idx = torch.arange(0, self.n_nodes).expand(self.n_nodes, -1)
        self.dist_per_idx = abs(col_idx - torch.t(col_idx))
        if self.scale_type == "pool":
            pool_adj = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(len(self.normalizer.pool_idx)):
                for d in self.normalizer.pool_idx[i]:
                    pool_adj = torch.diagonal_scatter(pool_adj, torch.tensor([i+1] * (self.n_nodes - d)), d)
            pool_adj_ = pool_adj[self.sub_row, self.sub_col]
            idx = torch.triu(torch.ones(self.n_good_nodes, self.n_good_nodes), diagonal = 1) == 1
            self.pool_adj_ut = pool_adj_[idx]
        

    def forward(self, 
                adj, 
                prop_adj = None):
        

        if self.good_bins is not None:
            adj_ = adj[self.sub_row, self.sub_col]
            if prop_adj is not None:
                prop_adj_ = prop_adj[self.sub_row, self.sub_col]
        else:
            adj_ = adj
            if prop_adj is not None:
                prop_adj_ = prop_adj

        
        
        z_mean, z_var = self.encoder(prop_adj_, adj_)
        
        if self.good_bins is not None:
            full_z_mean = torch.zeros(self.n_nodes, z_mean.shape[1]).double().to(self.device)
            full_z_mean[self.good_bins,:] = z_mean
        else:
            full_z_mean = z_mean
        
        if self.variational:

            
            
            z = z_mean + torch.randn_like(z_var) * z_var
            z_dist = Normal(z_mean, z_var)
            pz_dist = Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
            kl_divergence_z = torch.mean(torch.sum(kl(z_dist, pz_dist), 1)) / self.n_good_nodes
            #kl_loss = (0.5 / self.n_good_nodes) * torch.mean(torch.sum(1 + 2 * z_std - torch.square(z_mean) - torch.square(torch.exp(z_std)), 1))
            #print('manual kl and aut kl are {} and {}'.format(kl_loss, kl_divergence_z))
        
        else:
            
            z = z_mean 
        
        full_z = torch.zeros(self.n_nodes, z.shape[1]).double().to(self.device)
        full_z[self.good_bins, :] = z
        
        idx = torch.triu(torch.ones(self.n_good_nodes, self.n_good_nodes), diagonal = 1) == 1
            
        unnorm_adj = self.decoder(full_z)
        norm_adj = self.normalizer.normalize(unnorm_adj, adj)
        
        norm_adj_ = norm_adj[self.sub_row, self.sub_col]
        ut_norm_adj = norm_adj_[idx]
        ut_norm_adj = torch.max(ut_norm_adj,
                                (torch.zeros(ut_norm_adj.shape) + 0.01).to(self.device))
        ut_adj = adj_[idx]
        #f = (self.pool_adj_ut > 2)
        if self.likelihood == "poisson":
            px = Poisson(ut_norm_adj)
        elif self.likelihood in ["nb", "zinb"]:
            if self.dispersion_type == "common":
                dispersions = torch.ones(ut_norm_adj.shape).to(self.device) * self.dispersion
            elif self.dispersion_type == "distance":
                dispersions = torch.max(self.dispersion,
                                        (torch.zeros(self.dispersion.shape).to(self.device)+ 0.01)).to(self.device)
                dispersion_mat = torch.gather(dispersions.expand(self.n_nodes,-1),
                                              1,
                                              self.dist_per_idx)[self.sub_row, self.sub_col]
                dispersions = dispersion_mat[idx]
            else:
                dispersions = self.dispersion

            dispersions = torch.max(dispersions,
                                    (torch.zeros(dispersions.shape).to(self.device) + 0.01))

            if self.likelihood == "nb":
                px = NegativeBinomial(mu = ut_norm_adj,
                                      theta = dispersions)
            else:
                library_size = ut_adj.sum()
                distances_dropouts = []
                for d in range(self.n_nodes):
                    dr_input = torch.Tensor([library_size, d]).double().to(self.device)
                    dist_dr = self.dropout_act(self.dropout_func(dr_input))
                    #dist_dr = self.dropout_act(self.dropout_func(torch.Tensor([library_size, d]).double()))
                    distances_dropouts.append(dist_dr)
                distances_dropouts = torch.Tensor(distances_dropouts).double()
                dropout_mat = torch.gather(distances_dropouts.expand(self.n_nodes,-1),
                                           1,
                                           self.dist_per_idx)[self.sub_row, self.sub_col]
                dropouts = dropout_mat[idx].to(self.device)
                px = ZeroInflatedNegativeBinomial(mu = ut_norm_adj,
                                                  theta = dispersions,
                                                  zi_logits = dropouts)
                px = ZeroInflatedNegativeBinomial(mu = ut_norm_adj,
                                                  theta = dispersions,
                                                  zi_logits = dropouts)
        recon_loss = -px.log_prob(ut_adj).sum(-1)
            
        
        if self.variational:
            return recon_loss + kl_divergence_z, full_z_mean, torch.exp(z_var), norm_adj, unnorm_adj
        else:
            return recon_loss, z_mean, 

    def get_recon_adj(self,
                      adj, 
                      prop_adj = None):

        adj_ = adj[self.sub_row, self.sub_col]
        if prop_adj is not None:
            prop_adj_ = prop_adj[self.sub_row, self.sub_col]
        
        z_mean, z_var = self.encoder(prop_adj_, adj_)
        
        
        if self.variational:
            z = z_mean + torch.randn_like(z_var) * z_var
        else:
            z = z_mean 
            
        full_z = torch.zeros(self.n_nodes, z.shape[1]).double().to(self.device)
        full_z[self.good_bins,:] = z
    
        idx = torch.triu(torch.ones(self.n_good_nodes, self.n_good_nodes), diagonal = 1) == 1
        
        unnorm_adj = self.decoder(full_z)
        norm_adj = self.normalizer.normalize(unnorm_adj, adj)
        return norm_adj, unnorm_adj, z

        