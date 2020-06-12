"""Implements cmu motion data loader."""

import numpy as np
import torch
from torch.utils.data import Dataset
from dnri.utils import data_utils

import argparse, glob, queue
import dnri.datasets.utils.acm_parser as acm_parser


class CmuMotionData(Dataset):
    def __init__(self, name, data_path, mode, params, test_full=False, mask_ind_file=None):
        self.name = name
        self.data_path = data_path
        self.mode = mode
        self.params = params
        self.train_data_len = params.get('train_data_len', -1)
        # Get preprocessing stats.
        loc_max, loc_min, vel_max, vel_min = self._get_normalize_stats()
        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min
        self.test_full = test_full

        # Load data.
        self._load_data()
        self.expand_train = params.get('expand_train', False)
        if self.mode == 'train' and self.expand_train and self.train_data_len > 0:
            self.all_inds = []
            for ind in range(len(self.feat)):
                t_ind = 0
                while t_ind < len(self.feat[ind]):
                    self.all_inds.append((ind, t_ind))
                    t_ind += self.train_data_len
        else:
            self.expand_train = False

    def __getitem__(self, index):
        if self.expand_train:
            ind, t_ind = self.all_inds[index]
            start_ind = np.random.randint(t_ind, t_ind + self.train_data_len)

            feat = self.feat[ind][start_ind:start_ind + self.train_data_len]
            if len(feat) < self.train_data_len:
                feat = self.feat[ind][-self.train_data_len:]
            return {'inputs':feat}
        else: 
            inputs = self.feat[index]
            size = len(inputs)
            if self.mode == 'train' and self.train_data_len > 0 and size > self.train_data_len:
                start_ind = np.random.randint(0, size-self.train_data_len)
                inputs = inputs[start_ind:start_ind+self.train_data_len]
            result = {'inputs': inputs}
        return result

    def __len__(self, ):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return len(self.feat)

    def _get_normalize_stats(self,):
        train_loc = np.load(self._get_npy_path('loc', 'train'), allow_pickle=True)
        train_vel = np.load(self._get_npy_path('vel', 'train'), allow_pickle=True)
        try:
            train_loc.max()
            self.dynamic_len = False
        except:
            self.dynamic_len = True
        if self.dynamic_len:
            max_loc = max(x.max() for x in train_loc)
            min_loc = min(x.min() for x in train_loc)
            max_vel = max(x.max() for x in train_vel)
            min_vel = min(x.min() for x in train_vel)
            return max_loc, min_loc, max_vel, min_vel
        else:
            return train_loc.max(), train_loc.min(), train_vel.max(), train_vel.min()

    def _load_data(self, ):
        #print('***Experiment hack: evaling on training.***')
        # Load data
        self.loc_feat = np.load(self._get_npy_path('loc', self.mode), allow_pickle=True)
        self.vel_feat = np.load(self._get_npy_path('vel', self.mode), allow_pickle=True)
        #self.edge_feat = np.load(self._get_npy_path('edges', self.mode))

        # Perform preprocessing.
        if self.dynamic_len:
            self.loc_feat = [data_utils.normalize(feat, self.loc_max, self.loc_min) for feat in self.loc_feat]
            self.vel_feat = [data_utils.normalize(feat, self.vel_max, self.vel_min) for feat in self.vel_feat]
            self.feat = [np.concatenate([loc_feat, vel_feat], axis=-1) for loc_feat, vel_feat in zip(self.loc_feat, self.vel_feat)]
            self.feat = [torch.from_numpy(np.array(feat, dtype=np.float32)) for feat in self.feat]
            print("FEATURE LEN: ",len(self.feat))
        else:
            self.loc_feat = data_utils.normalize(
                self.loc_feat, self.loc_max, self.loc_min)
            self.vel_feat = data_utils.normalize(
                self.vel_feat, self.vel_max, self.vel_min)

            # Reshape [num_sims, num_timesteps, num_agents, num_dims]
            #self.loc_feat = np.transpose(self.loc_feat, [0, 1, 3, 2])
            #self.vel_feat = np.transpose(self.vel_feat, [0, 1, 3, 2])
            self.feat = np.concatenate([self.loc_feat, self.vel_feat], axis=-1)

            # Convert to pytorch cuda tensor.
            self.feat = torch.from_numpy(
                np.array(self.feat, dtype=np.float32))  # .cuda()

            # Only extract the first 49 frame if testing.
            if self.mode == 'test' and not self.test_full:
                self.feat = self.feat[:, :49]

    def _get_npy_path(self, feat, mode):
        return '%s/%s_%s_%s.npy' % (self.data_path,
                                    feat,
                                    mode,
                                    self.name)


def generate_all_loc_vel_np(all_np_motion, num_train, num_val, seq_len, mode='train', trial_inds=None):
    if trial_inds is not None:
        subset_np_motion = [all_np_motion[i] for i in trial_inds]
    elif mode == 'train':
        subset_np_motion = all_np_motion[:num_train]
    elif mode == 'valid':
        subset_np_motion = all_np_motion[num_train:num_train+num_val]
    elif mode == 'test':
        subset_np_motion = all_np_motion[num_train+num_val:]
    ret_loc_all = []
    ret_vel_all = []
    for np_motion in subset_np_motion:
        ret_loc, ret_vel = generate_loc_vel_np(np_motion, seq_len)
        ret_loc_all.append(ret_loc)
        ret_vel_all.append(ret_vel)
    if seq_len != -1:
        ret_loc_all = np.concatenate(ret_loc_all)
        ret_vel_all = np.concatenate(ret_vel_all)
    return ret_loc_all, ret_vel_all


def generate_loc_vel_np(np_motion, seq_len=50):
    loc = np_motion[1:]
    vel = np_motion[1:] - np_motion[:-1]
    if seq_len == -1:
        return loc, vel
    ret_loc = []
    ret_vel = []
    for k in range(0, np_motion.shape[0]-seq_len, seq_len):
        ret_loc.append(np.expand_dims(loc[k:k+seq_len], 0))
        ret_vel.append(np.expand_dims(vel[k:k+seq_len], 0))
    ret_loc = np.concatenate(ret_loc, 0)
    ret_vel = np.concatenate(ret_vel, 0)
    return ret_loc, ret_vel


def process_all_amc_file(data_path):
    all_amc_file = sorted(glob.glob(data_path+'*.amc'))
    asf_path = glob.glob(data_path + '*.asf')[0]
    all_np_motion = []
    total_frame = 0
    final_joint_masks = None
    edges = []
    joint_ids = {}
    joints = acm_parser.parse_asf(asf_path)
    for joint_idx, joint_name in enumerate(joints):
        joint_ids[joint_name] = joint_idx
    for joint_idx, joint_name in enumerate(joints):
        joint = joints[joint_name]
        for child in joint.children:
            edges.append([joint_ids[joint_name], joint_ids[child.name]])

    for amc_path in all_amc_file:
        label_alternating_joints(joints)
        motions = acm_parser.parse_amc(amc_path)
        np_motion, joint_masks = process_amc_file(joints, motions)
        if final_joint_masks is None:
            final_joint_masks = joint_masks
        else:
            assert np.array_equal(final_joint_masks, joint_masks)
        all_np_motion.append(np_motion)
        total_frame += np_motion.shape[0]
    return all_np_motion, final_joint_masks, edges


def process_amc_file(joints, motions):
    out_array = np.zeros((len(motions), len(joints), 3))
    joint_masks = np.zeros(len(joints))
    for frame_idx in range(len(motions)):
        joints['root'].set_motion(motions[frame_idx])
        for joint_idx, joint_name in enumerate(joints):
            c0, c1, c2 = joints[joint_name].coordinate
            out_array[frame_idx, joint_idx, 0] = c0
            out_array[frame_idx, joint_idx, 1] = c1
            out_array[frame_idx, joint_idx, 2] = c2
    for joint_idx, joint_name in enumerate(joints):
        joint_masks[joint_idx] = float(joints[joint_name].is_labeled)
    return out_array, joint_masks


def label_alternating_joints(joints):
    root = joints['root']
    root.is_labeled = False
    all_children = queue.Queue()
    for child in root.children:
        all_children.put(child)
    while not all_children.empty():
        child = all_children.get()
        child.is_labeled = not child.parent.is_labeled
        for new_child in child.children:
            all_children.put(new_child)


def load_trial_list(trial_list_file):
    # NOTE: this assumes files are consecutively labeled from 1 to num_trials
    with open(trial_list_file, 'r') as fin:
        data = fin.readlines()
    train_trials = np.array([int(x) for x in data[0].strip().split()]) - 1
    print("TRAIN TRIALS: ",train_trials.dtype)
    val_trials = np.array([int(x) for x in data[1].strip().split()]) - 1
    test_trials = np.array([int(x) for x in data[2].strip().split()]) - 1
    return train_trials, val_trials, test_trials

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--num_train_trials', type=int, default=12)
    parser.add_argument('--num_val_trials', type=int, default=4)
    parser.add_argument('--train_seq_len', type=int, default=50)
    parser.add_argument('--test_seq_len', type=int, default=100)
    parser.add_argument('--trial_list_file')
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.out_path
    if args.trial_list_file is not None:
        train_trials, val_trials, test_trials = load_trial_list(args.trial_list_file)
    else:
        train_trials, val_trials, test_trials = None, None, None
    all_np_motion, joint_masks, edges = process_all_amc_file(data_path)
    train_loc, train_vel = generate_all_loc_vel_np(all_np_motion, args.num_train_trials, args.num_val_trials, args.train_seq_len, mode='train', trial_inds=train_trials)
    valid_loc, valid_vel = generate_all_loc_vel_np(all_np_motion, args.num_train_trials, args.num_val_trials, args.train_seq_len, mode='valid', trial_inds=val_trials)
    test_loc, test_vel = generate_all_loc_vel_np(all_np_motion, args.num_train_trials, args.num_val_trials, args.test_seq_len, mode='test', trial_inds=test_trials)
    # Save train.
    #print(train_loc.shape)
    np.save(out_path + '%s_train_cmu.npy' % 'loc', train_loc)
    np.save(out_path + '%s_train_cmu.npy' % 'vel', train_vel)

    # Save valid
    #print(valid_loc.shape)
    np.save(out_path + '%s_valid_cmu.npy' % 'loc', valid_loc)
    np.save(out_path + '%s_valid_cmu.npy' % 'vel', valid_vel)

    # Save test
    #print(test_loc.shape)
    np.save(out_path + '%s_test_cmu.npy' % 'loc', test_loc)
    np.save(out_path + '%s_test_cmu.npy' % 'vel', test_vel)

    # Save joint masks
    np.save(out_path + 'joint_masks.npy', joint_masks)

    # Save edges
    np.save(out_path + 'edges.npy', edges)