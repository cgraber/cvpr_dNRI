from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dnri.utils import data_utils
import os
import numpy as np


def eval_forward_prediction(model, dataset, burn_in_steps, forward_pred_steps, params, return_total_errors=False):
    dataset.return_edges = False
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu)
    model.eval()
    total_se = 0
    batch_count = 0
    all_errors = []
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
            batch_count += 1
            if return_total_errors:
                all_errors.append(F.mse_loss(model_preds, gt_predictions, reduction='none').view(model_preds.size(0), model_preds.size(1), -1).mean(dim=-1))
            else:
                total_se += F.mse_loss(model_preds, gt_predictions, reduction='none').view(model_preds.size(0), model_preds.size(1), -1).mean(dim=-1).sum(dim=0)
    if return_total_errors:
        return torch.cat(all_errors, dim=0)
    else:
        return total_se / len(dataset)

def eval_forward_prediction_fixedwindow(model, dataset, burn_in_steps, forward_pred_steps, params, return_total_errors=False):
    dataset.return_edges = False
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    data_loader = DataLoader(dataset, batch_size=1)
    model.eval()
    total_se = 0
    batch_count = 0
    all_errors = []
    total_count = torch.zeros(forward_pred_steps)
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        print("BATCH IND %d OF %d"%(batch_ind+1, len(data_loader)))
        with torch.no_grad():

            if gpu:
                inputs = inputs.cuda(non_blocking=True)
            model_preds = model.predict_future_fixedwindow(inputs, burn_in_steps, forward_pred_steps, batch_size).cpu()
            for window_ind in range(model_preds.size(1)):
                current_preds = model_preds[:, window_ind]
                start_ind = burn_in_steps + window_ind
                gt_preds = inputs[:, start_ind:start_ind + forward_pred_steps].cpu()
                if gt_preds.size(1) < forward_pred_steps:
                    mask = torch.cat([torch.ones(gt_preds.size(1)), torch.zeros(forward_pred_steps - gt_preds.size(1))])
                    gt_preds = torch.cat([gt_preds, torch.zeros(gt_preds.size(0), forward_pred_steps-gt_preds.size(1), gt_preds.size(2), gt_preds.size(3))], dim=1)
                else:
                    mask = torch.ones(forward_pred_steps)
                total_se += F.mse_loss(current_preds, gt_preds, reduction='none').view(current_preds.size(0), current_preds.size(1), -1).mean(dim=-1).sum(dim=0).cpu()*mask
                total_count += mask

    return total_se / total_count