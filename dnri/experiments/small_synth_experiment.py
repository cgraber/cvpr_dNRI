from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.small_synth_data import SmallSynthData
import dnri.training.train as train
import dnri.training.train_utils as train_utils
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

def eval_edges(model, dataset, params):

    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    eval_metric = params.get('eval_metric')
    num_edge_types = params['num_edge_types']
    skip_first = params['skip_first']
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu)
    full_edge_count = 0.
    model.eval()
    correct_edges = 0.
    edge_count = 0.
    correct_0_edges = 0.
    edge_0_count = 0.
    correct_1_edges = 0.
    edge_1_count = 0.

    correct = num_predicted = num_gt = 0
    all_edges = []
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch['edges'].long()
        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                gt_edges = gt_edges.cuda(non_blocking=True)

            _, _, _, edges, _ = model.calculate_loss(inputs, is_train=False, return_logits=True)
            edges = edges.argmax(dim=-1)
            all_edges.append(edges.cpu())
            if len(edges.shape) == 3 and len(gt_edges.shape) == 2:
                gt_edges = gt_edges.unsqueeze(1).expand(gt_edges.size(0), edges.size(1), gt_edges.size(1))
            elif len(gt_edges.shape) == 3 and len(edges.shape) == 2:
                edges = edges.unsqueeze(1).expand(edges.size(0), gt_edges.size(1), edges.size(1))
            if edges.size(1) == gt_edges.size(1) - 1:
                gt_edges = gt_edges[:, :-1]
            edge_count += edges.numel()
            full_edge_count += gt_edges.numel()
            correct_edges += ((edges == gt_edges)).sum().item()
            edge_0_count += (gt_edges == 0).sum().item()
            edge_1_count += (gt_edges == 1).sum().item()
            correct_0_edges += ((edges == gt_edges)*(gt_edges == 0)).sum().item()
            correct_1_edges += ((edges == gt_edges)*(gt_edges == 1)).sum().item()
            correct += (edges*gt_edges).sum().item()
            num_predicted += edges.sum().item()
            num_gt += gt_edges.sum().item()
    prec = correct / (num_predicted + 1e-8)
    rec = correct / (num_gt + 1e-8)
    f1 = 2*prec*rec / (prec+rec+1e-6)
    all_edges = torch.cat(all_edges)
    return f1, correct_edges / (full_edge_count + 1e-8), correct_0_edges / (edge_0_count + 1e-8), correct_1_edges / (edge_1_count + 1e-8), all_edges

def plot_sample(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1)
    use_gt_edges = params.get('use_gt_edges')
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    batch_count = 0
    all_errors = []
    burn_in_steps = 10
    forward_pred_steps = 40
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch.get('edges', None)
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if gt_edges is not None and use_gt_edges:
                    gt_edges = gt_edges.cuda(non_blocking=True)
            if not use_gt_edges:
                gt_edges=None
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
            #total_se += F.mse_loss(model_preds, gt_predictions).item()
            print("MSE: ", torch.nn.functional.mse_loss(model_preds, gt_predictions).item())
            batch_count += 1
        fig, ax = plt.subplots()
        unnormalized_preds = dataset.unnormalize(model_preds)
        unnormalized_gt = dataset.unnormalize(inputs)
        def update(frame):
            ax.clear()
            ax.plot(unnormalized_gt[0, frame, 0, 0], unnormalized_gt[0, frame, 0, 1], 'bo')
            ax.plot(unnormalized_gt[0, frame, 1, 0], unnormalized_gt[0, frame, 1, 1], 'ro')
            ax.plot(unnormalized_gt[0, frame, 2, 0], unnormalized_gt[0, frame, 2, 1], 'go')
            if frame >= burn_in_steps:
                tmp_fr = frame - burn_in_steps
                ax.plot(unnormalized_preds[0, tmp_fr, 0, 0], unnormalized_preds[0, tmp_fr, 0, 1], 'bo', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 1, 0], unnormalized_preds[0, tmp_fr, 1, 1], 'ro', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 2, 0], unnormalized_preds[0, tmp_fr, 2, 1], 'go', alpha=0.5)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
        ani = animation.FuncAnimation(fig, update, interval=100, frames=burn_in_steps+forward_pred_steps)
        path = os.path.join(params['working_dir'], 'pred_trajectory_%d.mp4'%batch_ind)
        ani.save(path, codec='mpeg4')
        if batch_count >= num_samples:
            break




if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--same_data_norm', action='store_true')
    parser.add_argument('--no_data_norm', action='store_true')
    parser.add_argument('--error_out_name', default='prediction_errors_%dstep.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--test_burn_in_steps', type=int, default=10)
    parser.add_argument('--error_suffix')
    parser.add_argument('--subject_ind', type=int, default=-1)

    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = 3
    params['input_size'] = 4
    params['input_time_steps'] = 50
    params['nll_loss_type'] = 'gaussian'
    train_data = SmallSynthData(args.data_path, 'train', params)
    val_data = SmallSynthData(args.data_path, 'val', params)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
 
    elif args.mode == 'eval':
        test_data = SmallSynthData(args.data_path, 'test', params)
        forward_pred = 50 - args.test_burn_in_steps
        test_mse  = evaluate.eval_forward_prediction(model, test_data, args.test_burn_in_steps, forward_pred, params)
        path = os.path.join(args.working_dir, args.error_out_name%args.test_burn_in_steps)
        np.save(path, test_mse.cpu().numpy())
        test_mse_1 = test_mse[0].item()
        test_mse_15 = test_mse[14].item()
        test_mse_25 = test_mse[24].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP: ",test_mse_1)
        print("\t15 STEP: ",test_mse_15)
        print("\t25 STEP: ",test_mse_25)


        f1, all_acc, acc_0, acc_1, edges = eval_edges(model, val_data, params)
        print("Val Edge results:")
        print("\tF1: ",f1)
        print("\tAll predicted edge accuracy: ",all_acc)
        print("\tFirst Edge Acc: ",acc_0)
        print("\tSecond Edge Acc: ",acc_1)
        out_dir = os.path.join(args.working_dir, 'preds')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'encoder_edges.npy')
        np.save(out_path, edges.numpy())

        plot_sample(model, test_data, args.test_burn_in_steps, params)
    elif args.mode == 'record_predictions':
        model.eval()
        burn_in = args.test_burn_in_steps
        forward_pred = 50 - args.test_burn_in_steps
        test_data = SmallSynthData(args.data_path, 'test', params)
        if args.subject_ind == -1:
            val_data_loader = DataLoader(test_data, batch_size=params['batch_size'])
            all_predictions = []
            all_edges = []
            for batch_ind,batch in enumerate(val_data_loader):
                print("BATCH %d of %d"%(batch_ind+1, len(val_data_loader)))
                inputs = batch['inputs']
                if args.gpu:
                    inputs = inputs.cuda(non_blocking=True)
                with torch.no_grad():
                    predictions, edges = model.predict_future(inputs[:, :burn_in], forward_pred, return_edges=True, return_everything=True)
                    all_predictions.append(predictions)
                    all_edges.append(edges)
            if args.error_suffix is not None:
                out_path = os.path.join(args.working_dir, 'preds/', 'all_test_subjects_%s.npy'%args.error_suffix)
            else:
                out_path = os.path.join(args.working_dir, 'preds/', 'all_test_subjects.npy')

            predictions = torch.cat(all_predictions, dim=0)
            edges = torch.cat(all_edges, dim=0)

        else:
            data = test_data[args.subject_ind]
            inputs = data['inputs'].unsqueeze(0)
            if args.gpu:
                inputs = inputs.cuda(non_blocking=True)
            with torch.no_grad():
                predictions, edges = model.predict_future(inputs[:, :burn_in], forward_pred, return_edges=True, return_everything=True)
                predictions = predictions.squeeze(0)
                edges = edges.squeeze(0)
            out_path = os.path.join(args.working_dir, 'preds/', 'subject_%d.npy'%args.subject_ind)
        tmp_dir = os.path.join(args.working_dir, 'preds/')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        torch.save([predictions.cpu(), edges.cpu()], out_path)

 