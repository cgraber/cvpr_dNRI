import numpy as np
import torch
from torch.utils.data import Dataset
from dnri.utils import data_utils

import argparse, os


class SmallSynthData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'val_feats')
            edge_path = os.path.join(data_path, 'val_edges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')
        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)
        self.same_norm = params['same_data_norm']
        self.no_norm = params['no_data_norm']
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min)*2/(self.feat_max-self.feat_min) - 1
        else:
            self.loc_max = train_data[:, :, :, :2].max()
            self.loc_min = train_data[:, :, :, :2].min()
            self.vel_max = train_data[:, :, :, 2:].max()
            self.vel_min = train_data[:, :, :, 2:].min()
            self.feats[:,:,:, :2] = (self.feats[:,:,:,:2]-self.loc_min)*2/(self.loc_max - self.loc_min) - 1
            self.feats[:,:,:,2:] = (self.feats[:,:,:,2:]-self.vel_min)*2/(self.vel_max-self.vel_min)-1

    def unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        else:
            result1 = (data[:, :, :, :2] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            result2 = (data[:, :, :, 2:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
            return np.concatenate([result1, result2], axis=-1)


    def __getitem__(self, idx):
        return {'inputs': self.feats[idx], 'edges':self.edges[idx]}

    def __len__(self):
        return len(self.feats)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=100)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--num_time_steps', type=int, default=50)
    parser.add_argument('--pull_factor', type=float, default=0.1)
    parser.add_argument('--push_factor', type=float, default=0.05)

    args = parser.parse_args()
    np.random.seed(1)
    all_data = []
    all_edges = []
    num_sims = args.num_train + args.num_val + args.num_test
    flip_count = 0
    total_steps = 0
    for sim in range(num_sims):
        p1_loc = np.random.uniform(-2, -1, size=(2))
        p1_vel = np.random.uniform(0.05, 0.1, size=(2))
        p2_loc = np.random.uniform(1, 2, size=(2))
        p2_vel = np.random.uniform(-0.05, -0.1, size=(2))
        p3_loc = np.random.uniform(-1, 1, size=(2))
        p3_vel = np.random.uniform(-0.05, 0.05, size=(2))

        current_feats = []
        current_edges = []
        for time_step in range(args.num_time_steps):
            current_edge = np.array([0,0,0,0,0,0])
            current_edges.append(current_edge)
            if np.linalg.norm(p3_loc - p1_loc) < 1:
                norm = np.linalg.norm(p3_loc - p1_loc)
                coef = 1 - norm
                dir_1 = (p3_loc - p1_loc)/norm
                p3_vel += args.push_factor*coef*dir_1
                current_edge[1] = 1
            if np.linalg.norm(p3_loc - p2_loc) < 1:
                norm = np.linalg.norm(p3_loc - p2_loc)
                coef = 1 - norm
                dir_2 = (p3_loc - p2_loc)/norm
                p3_vel += args.push_factor*coef*dir_2
                current_edge[3] = 1

            p1_loc += p1_vel
            p2_loc += p2_vel
            p3_loc += p3_vel
            p1_feat = np.concatenate([p1_loc, p1_vel])
            p2_feat = np.concatenate([p2_loc, p2_vel])
            p3_feat = np.concatenate([p3_loc, p3_vel])
            new_feat = np.stack([p1_feat, p2_feat, p3_feat])
            current_feats.append(new_feat)
        all_data.append(np.stack(current_feats))
        all_edges.append(np.stack(current_edges))
        
    all_data = np.stack(all_data)
    train_data = torch.FloatTensor(all_data[:args.num_train])
    val_data = torch.FloatTensor(all_data[args.num_train:args.num_train+args.num_val])
    test_data = torch.FloatTensor(all_data[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_feats')
    torch.save(train_data, train_path)
    val_path = os.path.join(args.output_dir, 'val_feats')
    torch.save(val_data, val_path)
    test_path = os.path.join(args.output_dir, 'test_feats')
    torch.save(test_data, test_path)

    train_edges = torch.FloatTensor(all_edges[:args.num_train])
    val_edges = torch.FloatTensor(all_edges[args.num_train:args.num_train+args.num_val])
    test_edges = torch.FloatTensor(all_edges[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_edges')
    torch.save(train_edges, train_path)
    val_path = os.path.join(args.output_dir, 'val_edges')
    torch.save(val_edges, val_path)
    test_path = os.path.join(args.output_dir, 'test_edges')
    torch.save(test_edges, test_path)
