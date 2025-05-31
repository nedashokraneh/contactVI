import os
import sys
from tqdm import trange
from typing import Literal

import torch
import cooler
import random 
import numpy as np
import pandas as pd
import anndata as ad
from random import sample
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

class scHiCDataset(Dataset):
    
    def __init__(self,
                 AnnData_path,
                 resolution,
                 chromosome_size_filepath,
                 chromosome,
                 preload_prop_maps = False,
                 include_diag = False,
                 training_ratio = 0.9,
                 GCN_version: Literal["count", "binary"] = "binary",
                 GCN_linear_nbr = True,
                 output_version: Literal["count", "binary"] = "count",
                 sample = None):
        
        self.AnnData_path = AnnData_path
        self.resolution = resolution 
        self.chromosome = chromosome
        chromosome_sizes = read_chr_sizes(chromosome_size_filepath)
        self.chromosome_size = int(np.ceil(chromosome_sizes[self.chromosome] / self.resolution))
        self.preload = preload_prop_maps
        self.include_diag = include_diag
        self.bad_cells = []
        self.GCN_version = GCN_version
        self.GCN_linear_nbr = GCN_linear_nbr
        self.output_version = output_version
        self.sample = sample
        self.adata = ad.read_h5ad(self.AnnData_path)
        self.adata = self.adata[:, self.adata.var["chrom"] == self.chromosome].copy()
        if not self.include_diag:
            self.adata = self.adata[:, self.adata.var["bin1_id"] != self.adata.var["bin2_id"]].copy()

        bin1_idx, bin2_idx = self.adata.var["bin1_id"].tolist(), self.adata.var["bin2_id"].tolist()
        self.graph_shape = torch.Size((self.chromosome_size, self.chromosome_size))
        if self.GCN_linear_nbr:
            unique_bins = np.unique(bin1_idx)
            linear_nbr_idx = torch.tensor([unique_bins.tolist() + (unique_bins + 1).tolist(), (unique_bins + 1).tolist() + unique_bins.tolist()])
            linear_nbr_diag = torch.tensor([1] * linear_nbr_idx.shape[1])
            self.linear_nbr_map = torch.sparse.FloatTensor(linear_nbr_idx, linear_nbr_diag, self.graph_shape)

        diag_idx = torch.tensor(np.arange(self.chromosome_size)).repeat(2).reshape(2, -1)
        diag_vec = torch.tensor([1] * self.chromosome_size)
        self.eye = torch.sparse.FloatTensor(diag_idx, diag_vec, self.graph_shape)
            
        self.idx = torch.tensor([bin1_idx + bin2_idx, bin2_idx + bin1_idx])
            
        if self.sample is not None:
            if self.sample > self.adata.shape[0]:
                raise ValueError("sample size is greater than the number of available samples.")
            self.adata = self.adata[:self.sample, :]
        
        if self.preload:
            self.load_contact_maps()
        
        self.num_cells = self.adata.shape[0]
        self.training_ratio = training_ratio
        self.split_train_val()
        self.find_good_bins()
        self.make_bulk_map()
        

    def split_train_val(self):
        
        self.training_size = int(self.num_cells * self.training_ratio)
        shuffled_index = np.arange(self.num_cells)
        random.shuffle(shuffled_index)
        self.training_index = shuffled_index[: self.training_size]
        self.validation_index = shuffled_index[self.training_size:]
        
    def make_bulk_map(self):

        if isinstance(self.adata.X, np.ndarray):
            v = self.adata.X.sum(axis = 0).tolist()
        else:
            v = self.adata.X.toarray().sum(axis = 0).tolist()
        v = torch.tensor(v + v)
        nz_idx = (v != 0)
        i_ = self.idx[:, nz_idx]
        v_ = v[nz_idx]
        self.bulk_map = torch.sparse.FloatTensor(i_, v_, self.graph_shape).to_dense()
        
        
    def load_contact_maps(self):
        
        
        self.contact_maps = []
        self.GCN_prop_maps = []
        graph_shape = torch.Size((self.chromosome_size, self.chromosome_size))
        self.bulk_map = torch.zeros(self.graph_shape)
        for cell_id in trange(self.adata.shape[0]):
            GCN_prop, contact_map = self.get_prop_and_out(cell_id)
            self.contact_maps.append(contact_map)
            self.GCN_prop_maps.append(GCN_prop)

    def get_prop_and_out(self, cell_id):

        if isinstance(self.adata.X, np.ndarray):
            v = self.adata.X[cell_id, :].tolist()
        else:
            v = self.adata.X[cell_id, :].toarray()[0].tolist()
        v = torch.tensor(v + v)
        nz_idx = (v != 0)
        i_ = self.idx[:, nz_idx]
        v_ = v[nz_idx]
        contact_map = torch.sparse.FloatTensor(i_, v_, self.graph_shape)
        if self.GCN_version == "count":
            adj = contact_map
        else:
            adj = (contact_map.to_dense() > 0).float().to_sparse()
        
        if self.GCN_linear_nbr:
            adj = adj + self.linear_nbr_map
        adj_tilde = adj + self.eye
        rows_sum = torch.sparse.sum(adj_tilde, [0])
        indices = rows_sum.indices()[0].repeat(2).reshape(2, -1)
        values = torch.pow(rows_sum.values(), -0.5)
        D_tilde = torch.sparse.FloatTensor(indices, values, self.graph_shape)
        
        adj_tilde_norm = torch.sparse.mm(D_tilde, torch.sparse.mm(adj_tilde, D_tilde))
        
        if self.output_version == "binary":
            contact_map = (contact_map > 0).int()
        elif self.output_version == "count":
            contact_map = contact_map
        
        return adj_tilde_norm, contact_map

    def find_good_bins(self):
    
        self.good_bins = torch.tensor(np.union1d(self.adata.var["bin1_id"].unique(), self.adata.var["bin2_id"].unique()))

    def calculate_bandnorm_scale_factors(self):
        distances = (self.adata.var["bin2_id"] - self.adata.var["bin1_id"]).values
        #unique_distances = np.unique(distances)
        #unique_distances.sort()
        #cell_band_ls = np.zeros((self.adata.shape[0], len(unique_distances)))
        cell_band_ls = np.zeros((self.adata.shape[0], self.chromosome_size))
        for d in range(self.chromosome_size):
            d_idx = np.where(distances == d)[0]
            cell_band_ls[:, d] = np.sum(self.adata.X[:, d_idx].toarray(), axis = 1)
        self.cell_band_scale_factors = (cell_band_ls / cell_band_ls.mean(axis = 0))
        self.cell_band_scale_factors[np.isnan(self.cell_band_scale_factors)] = 0

    def calculate_cell_scale_factors(self):
        cell_ls = np.array(self.adata.X.sum(axis = 1)).flatten()
        self.cell_scale_factors = cell_ls / cell_ls.mean()

    def calculate_avg_pool_depth(self):
        self.pools_avg_depth = []
        curr_pool_size = 1
        curr_start_band = 1
        while curr_start_band <= (self.chromosome_size - 1):
            pool_idx = np.arange(curr_start_band, 
                                 np.min([curr_start_band + curr_pool_size, self.chromosome_size]),
                                 dtype = "int")
            cells_pool_sum = None
            for band_idx in pool_idx:
                band_adata = self.adata.X[:, (self.adata.var["bin2_id"] - self.adata.var["bin1_id"]) == band_idx].copy()
                if cells_pool_sum is None:
                    cells_pool_sum = band_adata.sum(axis = 1)
                else:
                    cells_pool_sum += band_adata.sum(axis = 1)
            self.pools_avg_depth.append(cells_pool_sum.mean())
            curr_start_band += curr_pool_size
            curr_pool_size += 1
        
        

def read_chr_sizes(chr_size_file):
    chr_sizes = {}
    with open(chr_size_file) as f:
        for line in f:
            chr_name, chr_size = line.rstrip("\n").split("\t")
            chr_sizes[chr_name] = int(chr_size)
    return chr_sizes
