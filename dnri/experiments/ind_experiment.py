from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.ind_data import IndData, ind_collate_fn
import dnri.training.train_dynamicvars as train
import dnri.training.train_utils as train_utils
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='val_prediction_errors.npy')
    parser.add_argument('--train_data_len', type=int, default=-1)
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--expand_train', action='store_true')
    parser.add_argument('--final_test', action='store_true')
    parser.add_argument('--test_short_sequences', action='store_true')

    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['input_size'] = 4
    params['nll_loss_type'] = 'gaussian'
    params['dynamic_vars'] = True
    params['collate_fn'] = ind_collate_fn
    train_data = IndData(args.data_path, 'train', params)
    val_data = IndData(args.data_path, 'valid', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        if args.final_test:
            test_data = IndData(args.data_path, 'test', params)
            test_mse, counts = evaluate.eval_forward_prediction_dynamicvars(model, test_data, params)
        else:
            test_mse, counts = evaluate.eval_forward_prediction_dynamicvars(model, val_data, params)
        path = os.path.join(args.working_dir, args.error_out_name)
        np.save(path, test_mse.cpu().numpy())
        path = os.path.join(args.working_dir, 'counts'+args.error_out_name)
        np.save(path, counts.cpu().numpy())
        test_mse_1 = test_mse[0].item()
        test_mse_20 = test_mse[19].item()
        test_mse_40 = test_mse[39].item()
        if args.final_test:
            print("TEST FORWARD PRED RESULTS:")
        else:
            print("VAL FORWARD PRED RESULTS:")
        print("\t1 STEP:  ", test_mse_1, counts[0].item())
        print("\t20 STEP: ", test_mse_20, counts[19].item())
        print("\t40 STEP: ", test_mse_40, counts[39].item())
        