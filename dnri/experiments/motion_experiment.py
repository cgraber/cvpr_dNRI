from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.cmu_motion_data import CmuMotionData
import dnri.training.train_utils as train_utils
import dnri.training.train as train
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='prediction_errors.npy')
    parser.add_argument('--train_data_len', type=int, default=-1)
    parser.add_argument('--expand_train', action='store_true')
    parser.add_argument('--fixed_window_len', type=int, default=40)
    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = 31
    params['input_noise_type'] = 'none'
    params['input_size'] = 6
    if args.train_data_len != -1:
        params['input_time_steps'] = args.train_data_len
    else:
        params['input_time_steps'] = 50
    params['nll_loss_type'] = 'gaussian'
    params['prior_variance'] = 5e-5
    name = 'cmu'
    train_data = CmuMotionData(name, args.data_path, 'train', params)
    val_data = CmuMotionData(name, args.data_path, 'valid', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode in ['eval', 'eval_masked', 'eval_fixedwindow'] or args.plot_prior_posterior:
        test_data = CmuMotionData(name, args.data_path, 'test', params, test_full=True)
        if args.mode == 'eval':
            test_cumulative_mse = evaluate.eval_forward_prediction(model, test_data, 50, 48, params)
            path = os.path.join(args.working_dir, args.error_out_name)
            np.save(path, test_cumulative_mse.cpu().numpy())
            test_mse_1 = test_cumulative_mse[0].item()
            test_mse_20 = test_cumulative_mse[19].item()
            test_mse_40 = test_cumulative_mse[39].item()
            print("FORWARD PRED RESULTS:")
            print("\t1 STEP:  ",test_mse_1)
            print("\t20 STEP: ", test_mse_20)
            print("\t40 STEP: ",test_mse_40)

        elif args.mode == 'eval_fixedwindow':
            print("RUNNING FIXED WINDOW EVAL")
            test_cumulative_mse = evaluate.eval_forward_prediction_fixedwindow(model, test_data, 50, args.fixed_window_len, params)
            path = os.path.join(args.working_dir, 'fixedwindow_' + args.error_out_name)
            np.save(path, test_cumulative_mse.cpu().numpy())
            test_mse_1 = test_cumulative_mse[0].item()
            test_mse_20 = test_cumulative_mse[19].item()
            test_mse_40 = test_cumulative_mse[39].item()
            print("FORWARD PRED RESULTS:")
            print("\t1 STEP:  ",test_mse_1)
            print("\t20 STEP: ", test_mse_20)
            print("\t40 STEP: ",test_mse_40)