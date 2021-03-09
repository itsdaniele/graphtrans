import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse, contains_self_loops, remove_self_loops
from scipy.linalg import eigh

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb

def no_aug(data):
    return data

def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (
        edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]]
                  for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    if len(edge_index) > 0:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    else: 
        data.edge_index = torch.tensor([[], []], dtype=torch.long)
    data.edge_attr = data.edge_attr[edge_mask]
    data.x = data.x[idx_nondrop]
    if hasattr(data, "node_depth"):
        data.node_depth = data.node_depth[idx_nondrop]

    return data


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_remain = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_remain]
    data.edge_attr = data.edge_attr[idx_remain]

    return data


def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = node_num - int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))
        idx_neigh -= set(idx_sub)

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array([n for n in range(edge_num) if (
        edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

    edge_index = data.edge_index.numpy()
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]]
                  for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    if len(edge_index) > 0:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    else:
        data.edge_index = torch.tensor([[], []], dtype=torch.long)
    data.x = data.x[idx_nondrop]
    data.edge_attr = data.edge_attr[edge_mask]
    if hasattr(data, "node_depth"):
        data.node_depth = data.node_depth[idx_nondrop]

    return data


def mask_nodes(data, aug_ratio):
    # TODO: This one need to be fixed for node attributes that are not encoded.
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    # token = torch.zeros_like(data.x[0], dtype=torch.long)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = 0

    return data

def mask_eigval(data, aug_ratio):
    # raise NotImplementedError
    num_nodes = data.num_nodes
    assert not contains_self_loops(data.edge_index)
    assert data.edge_attr is None
    
    edge_index, edge_weight = get_laplacian(data.edge_index, num_nodes=num_nodes)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
    # print("num_node", num_nodes)
    # print(edge_index.size())
    # print(edge_index)
    # print(L)

    eigval, eigvec = torch.symeig(L, eigenvectors=True)

    mask_num = int(num_nodes * aug_ratio)
    eigval_mask = np.random.choice(num_nodes, mask_num, replace=False)
    eigval[eigval_mask] = 0


    L = torch.matmul(eigvec, torch.matmul(eigval.diag_embed(), eigvec.transpose(-2, -1)))
    L.masked_fill_(L.abs().lt(0.05), 0)
    # print(L)

    edge_index = L.nonzero(as_tuple=False).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    # print(edge_index.size())
    # print(edge_index)
    # print(edge_index.size())
    # print(edge_index)
    # perm = idx_adj[edge_index[0], edge_index[1]] - 1
    # print(perm)
    # print((perm > 0).sum())
    
    data.edge_index = edge_index

    return data

AUGMENTATIONS = {
    "dnodes": drop_nodes,
    "pedges": permute_edges,
    "subgraph": subgraph,
    "mask_nodes": mask_nodes,
    "mask_eigval": mask_eigval,
    "none": no_aug
}

if __name__ == "__main__":
    from tqdm import tqdm
    from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    dataset = PygGraphPropPredDataset(name='ogbg-code', root='/data/zhwu/ogb')
    split_idx = dataset.get_idx_split()
    for data in tqdm(dataset[split_idx["train"]]):
        edge_index = data.edge_index
        # print(data.x)
        # mask = edge_index[0] == edge_index[1]
        # self_loop_num = mask.sum().item()
        # assert self_loop_num == 0, self_loop_num
        # Test
        mask_eigval(data, 0.01)
        break

