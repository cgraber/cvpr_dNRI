import numpy as np
import torch
from torch.utils.data import Dataset
from dnri.utils import data_utils


class BasketballData(Dataset):
	def __init__(self, name, data_path, mode, params, test_full=False, num_in_path=True, has_edges=True, transpose_data=True, max_len=None):
		self.name = name
		self.data_path = data_path
		self.mode = mode
		self.params = params
		self.num_in_path = num_in_path
		# Get preprocessing stats.
		loc_max, loc_min, vel_max, vel_min = self._get_normalize_stats()
		self.loc_max = loc_max
		self.loc_min = loc_min
		self.vel_max = vel_max
		self.vel_min = vel_min
		self.test_full = test_full
		self.max_len = max_len

		# Load data.
		self._load_data(transpose_data)
		
	def __getitem__(self, index):
		if self.max_len is not None:
			inputs = self.feat[index, :self.max_len]
		else:
			inputs = self.feat[index]
		return {'inputs': inputs}

	def __len__(self, ):
		return self.feat.shape[0]

	def _get_normalize_stats(self,):
		train_loc = np.load(self._get_npy_path('loc', 'train'))
		train_vel = np.load(self._get_npy_path('vel', 'train'))
		return train_loc.max(), train_loc.min(), train_vel.max(), train_vel.min()

	def _load_data(self, transpose_data):
		# Load data
		self.loc_feat = np.load(self._get_npy_path('loc', self.mode))
		self.vel_feat = np.load(self._get_npy_path('vel', self.mode))

		# Perform preprocessing.
		self.loc_feat = data_utils.normalize(
				self.loc_feat, self.loc_max, self.loc_min)
		self.vel_feat = data_utils.normalize(
				self.vel_feat, self.vel_max, self.vel_min)
		

		# Reshape [num_sims, num_timesteps, num_agents, num_dims]
		if transpose_data:
			self.loc_feat = np.transpose(self.loc_feat, [0, 1, 3, 2])
			self.vel_feat = np.transpose(self.vel_feat, [0, 1, 3, 2])
		self.feat = np.concatenate([self.loc_feat, self.vel_feat], axis=-1)

		# Convert to pytorch cuda tensor.
		self.feat = torch.from_numpy(
				np.array(self.feat, dtype=np.float32))  # .cuda()

		# Exlucde self edges.
		num_atoms = self.params['num_agents']
		off_diag_idx = np.ravel_multi_index(
				np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
				[num_atoms, num_atoms])

	def _get_npy_path(self, feat, mode):
		if self.num_in_path:
			return '%s/%s_%s_%s%s.npy' % (self.data_path,
										  feat,
										  mode,
										  self.name,
										  self.params['num_agents'])
		else:
			return '%s/%s_%s_%s.npy' % (self.data_path,
										feat,
										mode,
										self.name)