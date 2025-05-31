import os
import sys
sys.path.append("~/projects/contactVI/src")
import argparse
import dataloader
import model
import modules

    
import torch
import numpy as np
import anndata as ad
from tqdm import trange
from torch import nn


def main():

    
    parser = create_parser()
    args = parser.parse_args()
    if "seed" in args:
        torch.manual_seed(args.seed)
    preload = True
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    AnnData_path = args.input_file
    resolution = args.resolution
    chromosome_size_filepath = args.chromosome_size
    include_diag = False
    chr = args.chromosome
    chr_name = 'chr{}'.format(chr)
    if args.gpu:
        device = "cuda"
    else:
        device = 'cpu'
    hidden1 = args.hidden_size
    hidden2 = int(hidden1 / 2)
    GCN_version = args.gcn_version
    if "sample_size" not in args:
        args.sample_size = None
    dataset = dataloader.scHiCDataset(AnnData_path, 
                                      resolution, 
                                      chromosome_size_filepath, 
                                      chr_name, 
                                      preload_prop_maps = preload,
                                      include_diag = include_diag,
                                      GCN_version = GCN_version,
                                      GCN_linear_nbr = args.linear_nbr,
                                      output_version = "count",
                                      sample = args.sample_size
                                     )
   
    dataset.calculate_avg_pool_depth()
    num_nodes = dataset.chromosome_size
    features = torch.eye(num_nodes).to(device)
    ut_idx = np.triu_indices(len(dataset.good_bins), 1)
    
    likelihood = args.likelihood
    my_model = model.GVAE(num_nodes, 
                       hidden1, 
                       hidden2, 
                       good_bins = dataset.good_bins,
                       likelihood = likelihood,
                       dispersion = "common",
                       scale_type = args.scale_type,
                        distance_effect = True,
                       decoder_type = args.decoder_type,
                       variational = True,
                          var_eps = 0.001,
                      GCN1_act = nn.ReLU(),
                         self_adj_input = args.self_input,
                         device = device)
    
    my_model = my_model.double()
    my_model = my_model.to(device)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr)
    losses = []
    validation_losses = []
    for epoch in trange(args.epoch):
        epoch_losses = []
        epoch_validation_losses = []
        for idx in dataset.training_index:
            prop_adj =  dataset.GCN_prop_maps[idx].to_dense().to(device)
            out_adj = dataset.contact_maps[idx].to_dense().to(device)
            outs = my_model(adj = torch.Tensor.double(out_adj), prop_adj = torch.Tensor.double(prop_adj))
            my_model.zero_grad()
            loss = outs[0]
            epoch_losses.append(loss.detach().to('cpu').numpy().item())
            loss.backward()
            optimizer.step()
        
        for idx in dataset.validation_index:
            prop_adj =  dataset.GCN_prop_maps[idx].to_dense().to(device)
            out_adj = dataset.contact_maps[idx].to_dense().to(device)
            outs = my_model(adj = torch.Tensor.double(out_adj), prop_adj = torch.Tensor.double(prop_adj))
            loss = outs[0]
            epoch_validation_losses.append(loss.detach().to('cpu').numpy().item())
        losses.append(np.mean(epoch_losses))
        validation_losses.append(np.mean(epoch_validation_losses))
        
    with open(os.path.join(args.output_dir, "losses.txt"), "w") as o:
        for loss in losses:
            o.write("{}\n".format(loss))

    with open(os.path.join(args.output_dir, "validation_losses.txt"), "w") as o:
        for loss in validation_losses:
            o.write("{}\n".format(loss))


    recons = []
    imp_array = []
    for idx in range(dataset.num_cells):
        with torch.no_grad():
            prop_adj =  dataset.GCN_prop_maps[idx].to_dense().to(device)
            out_adj = dataset.contact_maps[idx].to_dense().to(device)
            scale_graph, *_ = my_model.get_recon_adj(adj = torch.Tensor.double(out_adj), prop_adj = torch.Tensor.double(prop_adj))
            scale_graph = scale_graph.cpu()
            imp_array.append(scale_graph.numpy()[np.ix_(dataset.good_bins, dataset.good_bins)][ut_idx].reshape(1, -1))
            recons.append(scale_graph.numpy() + scale_graph.numpy().T)
    
    imp_array = np.concatenate(imp_array, axis = 0)
    if imp_array.shape[0] != dataset.num_cells:
        imp_array = imp_array.reshape(dataset.num_cells, -1)
    np.save(os.path.join(args.output_dir, "imp"), imp_array)



def create_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input-file', action = 'store', required = True, 
                        help = 'The path of input anndata.')

    parser.add_argument('--sample-size', action = 'store', type = int,
                       help = "number of cells to use.")
    
    parser.add_argument('-l', '--chromosome-size', action = 'store', required = True, 
                        help = 'The path of chromosome size file.')
    
    parser.add_argument('-c', '--chromosome', action = 'store', required = True, type = int,
                        help = "chromosome to denoise."
                       )
    
    parser.add_argument('-r', '--resolution', action = 'store', required = True, type = int,
                        help = "resolution"
                       )

    parser.add_argument('--hidden-size', action = 'store', default = 32, type = int,
                       help = "size of the hidden layer")
    
    parser.add_argument('-d', '--decoder-type', action = 'store', required = True,
                        help = "decoder type"
                       )
    
    parser.add_argument('-s', '--scale-type', action = 'store', required = True,
                        help = "scale type"
                       )
    
    parser.add_argument('-g', '--gcn-version', action = 'store', required = True, 
                        help = "binary or count."
                       )

    parser.add_argument('--linear-nbr', action = 'store_true', default = False, 
                        help = "whether adding linear neighbors or not."
                       )
    
    parser.add_argument('--self-input', action = 'store_true', default = False, \
                        help = 'if set, contact maps are used as node features.', required = False)

    parser.add_argument('--gpu', action = 'store_true', default = False, \
                        help = 'if set, use GPU.', required = False)

    parser.add_argument('--likelihood', action = 'store', default = "poisson", \
                        help = 'distribution to calculate likelihood', required = False)
    
    parser.add_argument('-o', '--output-dir', action = 'store',
                        help = "The path to store results.")

    parser.add_argument('--seed', action = 'store', default = 0,
                        help = "random seed")

    parser.add_argument('--epoch', action = 'store', type = int, default = 60,
                        help = "number of epochs")

    parser.add_argument('--lr', action = 'store', type = float, default = 0.001,
                        help = "learning rate")
    
    return parser


if __name__ == "__main__":
    main()
