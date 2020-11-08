import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from . import train_utils
import dnri.utils.misc as misc

import time, os

import random
import numpy as np

def train(model, train_data, val_data, params, train_writer, val_writer):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    sub_batch_size = params.get('sub_batch_size')
    if sub_batch_size is None:
        sub_batch_size = batch_size
    val_batch_size = params.get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params.get('accumulate_steps', 1)
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 1)
    val_start = params.get('val_start', 0)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    normalize_nll = params.get('normalize_nll', False)
    normalize_kl = params.get('normalize_kl', False)
    tune_on_nll = params.get('tune_on_nll', False)
    verbose = params.get('verbose', False)
    val_teacher_forcing = params.get('val_teacher_forcing', False)
    collate_fn = params.get('collate_fn', None)
    continue_training = params.get('continue_training', False)
    normalize_inputs = params['normalize_inputs']
    num_decoder_samples = 1
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    print("NUM BATCHES: ",len(train_data_loader))
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size, collate_fn=collate_fn)
    lr = params['lr']
    wd = params.get('wd', 0.)
    mom = params.get('mom', 0.)
    
    model_params = [param for param in model.parameters() if param.requires_grad]
    if params.get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if continue_training:
        print("RESUMING TRAINING")
        model.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_epoch = train_params['best_val_epoch']
        model.steps = train_params['step']
        print("STARTING EPOCH: ",start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = 10000000
    
    training_scheduler = train_utils.build_scheduler(opt, params)
    end = start = 0 
    misc.seed(1)
    for epoch in range(start_epoch, num_epochs+1):
        model.epoch = epoch
        print("EPOCH", epoch, (end-start))

        model.train_percent = epoch / num_epochs
        start = time.time() 
        for batch_ind, batch in enumerate(train_data_loader):
            model.train()
            inputs = batch['inputs']
            masks = batch.get('masks', None)
            node_inds = batch.get('node_inds', None)
            graph_info = batch.get('graph_info', None)
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                if masks is not None:
                    masks = masks.cuda(non_blocking=True)
            args = {'is_train':True, 'return_logits':True}
            sub_steps = len(range(0, batch_size, sub_batch_size))
            for sub_batch_ind in range(0, batch_size, sub_batch_size):
                sub_inputs = inputs[sub_batch_ind:sub_batch_ind+sub_batch_size]
                for sample in range(num_decoder_samples):
                    if normalize_inputs:
                        if masks is not None:
                            normalized_inputs = model.normalize_inputs(inputs[:, :-1], masks[:, :-1])
                        else:
                            normalized_inputs = model.normalize_inputs(inputs[:, :-1])
                        args['normalized_inputs'] = normalized_inputs[sub_batch_ind:sub_batch_ind+sub_batch_size]
                    if masks is not None:
                        sub_masks = masks[sub_batch_ind:sub_batch_ind+sub_batch_size]
                        sub_node_inds = node_inds[sub_batch_ind:sub_batch_ind+sub_batch_size]
                        sub_graph_info = graph_info[sub_batch_ind:sub_batch_ind+sub_batch_size]
                        loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(sub_inputs, sub_masks, sub_node_inds, sub_graph_info, **args)
                    else:
                        loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(sub_inputs, **args)
                    loss = loss / (sub_steps*accumulate_steps*num_decoder_samples)
                    loss.backward()
                
                if verbose:
                    tmp_batch_ind = batch_ind*sub_steps + sub_batch_ind + 1
                    tmp_total_batch = len(train_data_loader)*sub_steps
                    print("\tBATCH %d OF %d: %f, %f, %f"%(tmp_batch_ind, tmp_total_batch, loss.item(), loss_nll.mean().item(), loss_kl.mean().item()))
            if accumulate_steps == -1 or (batch_ind+1)%accumulate_steps == 0:
                if verbose and accumulate_steps > 0:
                    print("\tUPDATING WEIGHTS")
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                elif clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)        
                opt.step()
                model.steps += 1
                opt.zero_grad()
                if accumulate_steps > 0 and accumulate_steps > len(train_data_loader) - batch_ind - 1:
                    break
            
        if training_scheduler is not None:
            training_scheduler.step()
        
        if train_writer is not None:
            train_writer.add_scalar('loss', loss.item()*(sub_steps*accumulate_steps*num_decoder_samples), global_step=epoch)
            if normalize_nll:
                train_writer.add_scalar('NLL', loss_nll.mean().item(), global_step=epoch)
            else:
                train_writer.add_scalar('NLL', loss_nll.mean().item()/(inputs.size(1)*inputs.size(2)), global_step=epoch)
            
            train_writer.add_scalar("KL Divergence", loss_kl.mean().item(), global_step=epoch)
        if ((epoch+1)%val_interval != 0):
            end = time.time()
            continue
        model.eval()
        opt.zero_grad()
        total_nll = 0
        total_kl = 0
        if verbose:
            print("COMPUTING VAL LOSSES")
        with torch.no_grad():
            for batch_ind, batch in enumerate(val_data_loader):
                inputs = batch['inputs']
                masks = batch.get('masks', None)
                node_inds = batch.get('node_inds', None)
                graph_info = batch.get('graph_info', None)
                if gpu:
                    inputs = inputs.cuda(non_blocking=True)
                    if masks is not None:
                        masks = masks.cuda(non_blocking=True)
                if masks is not None:
                    loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(inputs, masks, node_inds, graph_info, is_train=False, teacher_forcing=val_teacher_forcing, return_logits=True)
                else:
                    loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(inputs, is_train=False, teacher_forcing=val_teacher_forcing, return_logits=True)
                total_kl += loss_kl.sum().item()
                total_nll += loss_nll.sum().item()
            if verbose:
                print("\tVAL BATCH %d of %d: %f, %f"%(batch_ind+1, len(val_data_loader), loss_nll.mean(), loss_kl.mean()))
            
        total_kl /= len(val_data)
        total_nll /= len(val_data)
        total_loss = model.kl_coef*total_kl + total_nll #TODO: this is a thing you fixed
        #total_loss = total_kl + total_nll
        if val_writer is not None:
            val_writer.add_scalar('loss', total_loss, global_step=epoch)
            val_writer.add_scalar("NLL", total_nll, global_step=epoch)
            val_writer.add_scalar("KL Divergence", total_kl, global_step=epoch)
        
        if tune_on_nll:
            tuning_loss = total_nll
        else:
            tuning_loss = total_loss
        if tuning_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = tuning_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        model.save(checkpoint_dir)
        torch.save({
                    'epoch':epoch+1,
                    'optimizer':opt.state_dict(),
                    'best_val_result':best_val_result,
                    'best_val_epoch':best_val_epoch,
                    'step':model.steps,
                   }, training_path)
        print("EPOCH %d EVAL: "%epoch)
        print("\tCURRENT VAL LOSS: %f"%tuning_loss)
        print("\tBEST VAL LOSS:    %f"%best_val_result)
        print("\tBEST VAL EPOCH:   %d"%best_val_epoch)
        
        end = time.time()

    
