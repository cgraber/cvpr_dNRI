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


def eval_forward_prediction_dynamicvars(model, dataset, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    collate_fn = params.get('collate_fn', None)
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=gpu, collate_fn=collate_fn)
    model.eval()
    total_se = 0
    batch_count = 0
    final_errors = torch.zeros(0)
    final_counts = torch.zeros(0)
    bad_count = 0
    for batch_ind, batch in enumerate(data_loader):
        print("DATA POINT ",batch_ind)
        inputs = batch['inputs']
        gt_preds = inputs[0, 1:]
        masks = batch['masks']
        node_inds = batch.get('node_inds', None)
        graph_info = batch.get('graph_info', None)
        burn_in_masks = batch['burn_in_masks']
        pred_masks = (masks.float() - burn_in_masks)[0, 1:]
        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                burn_in_masks = burn_in_masks.cuda(non_blocking=True)
            model_preds = model.predict_future(inputs, masks, node_inds, graph_info, burn_in_masks)[0].cpu()
            max_len = pred_masks.sum(dim=0).max().int().item()
            if max_len > len(final_errors):
                final_errors = torch.cat([final_errors, torch.zeros(max_len - len(final_errors))])
                final_counts = torch.cat([final_counts, torch.zeros(max_len - len(final_counts))])
            for var in range(masks.size(-1)):
                var_gt = gt_preds[:, var]
                var_preds = model_preds[:, var]
                var_pred_masks = pred_masks[:, var]
                var_losses = F.mse_loss(var_preds, var_gt, reduction='none').mean(dim=-1)*var_pred_masks
                tmp_inds = torch.nonzero(var_pred_masks)
                if len(tmp_inds) == 0:
                    continue
                for i in range(len(tmp_inds)-1):
                    if tmp_inds[i+1] - tmp_inds[i] != 1:
                        bad_count += 1
                        break
                num_entries = var_pred_masks.sum().int().item()
                final_errors[:num_entries] += var_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
                final_counts[:num_entries] += var_pred_masks[tmp_inds[0]:tmp_inds[0]+num_entries]
    print("FINAL BAD COUNT: ",bad_count)
    return final_errors/final_counts, final_counts