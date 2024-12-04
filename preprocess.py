import os
import numpy as np
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import ToDense
from functools import partial
from multiprocessing import Pool
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm
import argparse
from utils import gen_dist_mask_pq_walk, pq_walk_sequences

splits = ['train', 'val', 'test']
metadata = {
    'MNIST': {'max_num_nodes': 75},
    'CIFAR10': {'max_num_nodes': 150},
    'PATTERN': {'max_num_nodes': 188},
    'CLUSTER': {'max_num_nodes': 190},
    'ZINC': {'max_num_nodes': 38}
}

func_sp = partial(floyd_warshall, directed=False, unweighted=True)
func_sp_ours = partial(gen_dist_mask_pq_walk, p=1, q=1, walk_length=10, num_walks=5)

def process(name):
    for split in splits:
        dataset = GNNBenchmarkDataset(root='./data', name=name, split=split,
                                      transform=ToDense(num_nodes=metadata[name]['max_num_nodes']))
        keys = dataset[0].keys()
        dataset_as_dict = {key: [] for key in keys}
        for g in dataset:
            for key in keys:
                dataset_as_dict[key].append(g[key].numpy())
        adjs = dataset_as_dict.pop('adj')
        # with Pool(25) as p: #! adjust according to your machine
        #     dist = p.map(func_sp, adjs)
        # dist = np.stack(dist)
        # dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
        # dist_mask = np.stack([(dist == k) for k in range(dist.max() + 1)], axis=1)
        
        dist_masks = []
        for adj in tqdm(adjs):
            dist_masks.append(func_sp_ours(adj))
        dist_mask = np.stack(dist_masks)
        
        if name in ['MNIST', 'CIFAR10']:
            np.savez(f'./data/{name}/{split}.npz',
                     x = np.concatenate([
                         np.stack(dataset_as_dict['x']),
                         np.stack(dataset_as_dict['pos'])
                     ], axis=-1),
                     y = np.concatenate(dataset_as_dict['y']).astype(np.int32),
                     node_mask = np.stack(dataset_as_dict['mask']))
        elif name in ['PATTERN', 'CLUSTER']:
            np.savez(f'./data/{name}/{split}.npz',
                     x = np.stack(dataset_as_dict['x']),
                     y = np.stack(dataset_as_dict['y']).astype(np.int32),
                     node_mask = np.stack(dataset_as_dict['mask']))
        np.save(f'./data/{name}/{split}_dist_mask', dist_mask)

def process_zinc():
    if not os.path.exists('./data/ZINC'):
        os.mkdir('./data/ZINC')
    for split in splits:
        dataset = ZINC(root='./data/ZINC', subset=True, split=split,
                       transform=ToDense(num_nodes=metadata['ZINC']['max_num_nodes']))
        keys = dataset[0].keys()
        dataset_as_dict = {key: [] for key in keys}
        for g in dataset:
            for key in keys:
                dataset_as_dict[key].append(g[key].numpy())
        
        adjs = dataset_as_dict.pop('adj')
        # with Pool(25) as p:
        #     dist = p.map(func_sp, adjs)
        # dist = np.stack(dist)
        # dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
        # dist_mask = np.stack([(dist == k) for k in range(dist.max() + 1)], axis=1)
        
        dist_masks = []
        for adj in tqdm(adjs):
            dist_masks.append(func_sp_ours(adj))
        dist_mask = np.stack(dist_masks)

        np.savez(f'./data/ZINC/subset/{split}.npz',
                 x = np.stack(dataset_as_dict['x']).squeeze().astype(np.int32),
                 y = np.concatenate(dataset_as_dict['y']),
                 node_mask = np.stack(dataset_as_dict['mask']))
        np.save(f'./data/ZINC/subset/{split}_dist_mask', dist_mask)
        np.save(f'./data/ZINC/subset/{split}_edge_attr', np.stack(adjs).astype(np.int32))


parser = argparse.ArgumentParser(description="Setting the lenght and number of the random walks")
parser.add_argument('--length', type=int, action='store_const', help="the length of the random walks", default=10)
parser.add_argument('--num', type=int, action='store_const', help="the length of the random walks", default=5)
parser.add_argument('--p_value', type=float, action='store_const', help="the length of the random walks", default=1.0)
parser.add_argument('--q_value', type=float, action='store_const', help="the length of the random walks", default=1.0)
args = parser.parse_args()

if __name__ == '__main__':

    walk_length = args.length
    num_walks = args.num
    p = args.p_value
    q = args.q_value
    func_sp_ours = partial(gen_dist_mask_pq_walk, p=p, q=q, walk_length=walk_length, num_walks=num_walks)

    if not os.path.exists('./data'):
        os.mkdir('./data')
    for name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        print(f'Processing {name}...')
        process(name)
        if os.path.exists('./data/'+ name + '/processed'):
            src_name = './data/'+ name + '/processed'
            os.rename(src_name, src_name + str(walk_length) + '-' + str(num_walks) + '-' + str(int(p * 100)) + '-' + str(int(q * 100)))
    print(f'Processing ZINC...')    
    process_zinc()
