from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.bball_data import BasketballData
import dnri.training.train as train
import dnri.training.train_utils as train_utils
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

import os
import numpy as np


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='prediction_errors.npy')
    parser.add_argument('--error_suffix')
    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = params['num_agents'] = 5
    params['input_noise_type'] = 'none'
    params['input_size'] = 4
    params['input_time_steps'] = 40
    params['nll_loss_type'] = 'gaussian'
    params['prior_variance'] = 5e-5
    name = 'bball'
    train_data = BasketballData(name, args.data_path, 'train', params, num_in_path=False, transpose_data=False, max_len=40)
    val_data = BasketballData(name, args.data_path, 'valid', params, num_in_path=False, transpose_data=False, max_len=40)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        test_data = BasketballData(name, args.data_path, 'test', params, num_in_path=False, transpose_data=False)
        test_cumulative_mse = evaluate.eval_forward_prediction(model, test_data, 40, 9, params)
        path = os.path.join(args.working_dir, args.error_out_name)
        np.save(path, test_cumulative_mse.cpu().numpy())
        test_mse_1 = test_cumulative_mse[0].item()
        test_mse_5 = test_cumulative_mse[4].item()
        test_mse_9 = test_cumulative_mse[8].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP:  ",test_mse_1)
        print("\t5 STEP: ", test_mse_5)
        print("\t9 STEP: ",test_mse_9)