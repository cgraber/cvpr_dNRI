import numpy as np


# Code from NRI.
def normalize(data, data_max, data_min):
	return (data - data_min) * 2 / (data_max - data_min) - 1


def unnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min


def get_edge_inds(num_vars):
	edges = []
	for i in range(num_vars):
		for j in range(num_vars):
			if i == j:
				continue
			edges.append([i, j])
	return edges