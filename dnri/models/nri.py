import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import model_utils


class BaseNRI(nn.Module):
    def __init__(self, num_vars, encoder, decoder, params):
        super(BaseNRI, self).__init__()
        # Model Params
        self.num_vars = num_vars
        self.decoder = decoder
        self.encoder = encoder
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        if params.get('no_edge_prior') is not None:
            prior = np.zeros(self.num_edge_types)
            prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
            prior[0] = params['no_edge_prior']
            log_prior = torch.FloatTensor(np.log(prior))
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = torch.unsqueeze(log_prior, 0)
            if params['gpu']:
                log_prior = log_prior.cuda(non_blocking=True)
            self.log_prior = log_prior
            print("USING NO EDGE PRIOR: ",self.log_prior)
        else:
            print("USING UNIFORM PRIOR")
            prior = np.zeros(self.num_edge_types)
            prior.fill(1.0/self.num_edge_types)
            log_prior = torch.FloatTensor(np.log(prior))
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = torch.unsqueeze(log_prior, 0)
            if params['gpu']:
                log_prior = log_prior.cuda(non_blocking=True)
            self.log_prior = log_prior

        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.extra_context = params.get('embedder_time_bins', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.no_prior = params.get('no_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False):
        # Should be shape [batch, num_edges, edge_dim]
        encoder_results = self.encoder(inputs)
        logits = encoder_results['logits']
        old_shape = logits.shape
        hard_sample = (not is_train) or self.train_hard_sample
        edges = model_utils.gumbel_softmax(
            logits.view(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        if not is_train and teacher_forcing:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        output = self.decoder(inputs[:, self.extra_context:-1], edges, 
                              teacher_forcing=teacher_forcing, 
                              teacher_forcing_steps=teacher_forcing_steps)
        if len(inputs.shape) == 4:
            target = inputs[:, self.extra_context+1:, :, :]
        else:
            target = inputs[:, self.extra_context+1:, :]
        loss_nll = self.nll(output, target)
        prob = F.softmax(logits, dim=-1)
        if self.no_prior:
            loss_kl = torch.cuda.FloatTensor([0.0])
        elif self.log_prior is not None:
            loss_kl = self.kl_categorical(prob)
        else:
            loss_kl = self.kl_categorical_uniform(prob)
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, logits, output
        else:
            return loss, loss_nll, loss_kl

        
    def predict_future(self, data_encoder, data_decoder):
        raise NotImplementedError()

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))

    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps) - self.log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)
    
    def kl_categorical_uniform(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds + eps) + np.log(self.num_edge_types))
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean()
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)/self.num_edge_types
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StaticNRI(BaseNRI):

    def copy_state(self, decoder_state):
        if isinstance(decoder_state, tuple) or isinstance(decoder_state, list):
            current_decoder_state = (decoder_state[0].clone(), decoder_state[1].clone())
        else:
            current_decoder_state = decoder_state.clone()
        return current_decoder_state

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        encoder_dict = self.encoder(inputs)
        logits = encoder_dict['logits']
        old_shape = logits.shape
        edges = nn.functional.gumbel_softmax(
            logits.view(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=True).view(old_shape)
        tmp_predictions, decoder_state = self.decoder(inputs[:, :-1], edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        decoder_inputs = inputs[:, -1].unsqueeze(1)
        predictions = self.decoder(decoder_inputs, edges, prediction_steps=prediction_steps, teacher_forcing=False, state=decoder_state)
        if return_everything:
            predictions = torch.cat([tmp_predictions, predictions], dim=1)
        if return_edges:
            return predictions, edges
        else:
            return predictions

    def merge_states(self, states):
        if isinstance(states[0], tuple) or isinstance(states[0], list):
            result0 = torch.cat([x[0] for x in states], dim=0)
            result1 = torch.cat([x[1] for x in states], dim=0)
            return (result0, result1)
        else:
            return torch.cat(states, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        burn_in_inputs = inputs[:, :burn_in_steps]
        encoder_dict = self.encoder(burn_in_inputs)
        logits = encoder_dict['logits']
        old_shape = logits.shape
        edges = nn.functional.gumbel_softmax(
            logits.view(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=True).view(old_shape)
        _, decoder_state = self.decoder(burn_in_inputs[:, :-1], edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        all_timestep_preds = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            encoder_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                tmp_ind = window_ind + step
                predictions = inputs[:, tmp_ind:tmp_ind+1]
                predictions, decoder_state = self.decoder(predictions,edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=decoder_state)
                current_batch_preds.append(predictions)
                decoder_states.append(self.copy_state(decoder_state))
            batch_decoder_state = self.merge_states(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            batch_edges = edges.expand(current_batch_preds.size(0), -1, -1)
            for step in range(prediction_steps - 1):
                current_batch_preds, batch_decoder_state =self.decoder(current_batch_preds, batch_edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=batch_decoder_state) 
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.cat(current_timestep_preds, dim=1))
        result = torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            return result.unsqueeze(0), edges
        else:
            return result.unsqueeze(0)

    
class DynamicNRI(BaseNRI):
    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        encoder_dict = self.encoder(inputs)
        burn_in_logits = encoder_dict['logits']
        encoder_state = encoder_dict['state']
        old_shape = burn_in_logits.shape
        burn_in_edges = nn.functional.gumbel_softmax(
            burn_in_logits.view(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=True).view(old_shape)
        burn_in_predictions, decoder_state = self.decoder(inputs, burn_in_edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        prev_preds = burn_in_predictions[:, -1].unsqueeze(1)
        preds = [prev_preds]
        all_edges = [burn_in_edges]
        for step in range(prediction_steps-1):
            encoder_dict = self.encoder(prev_preds, encoder_state)
            logits = encoder_dict['logits']
            encoder_state = encoder_dict['state']
            old_shape = logits.shape
            edges = nn.functional.gumbel_softmax(
                logits.view(-1, self.num_edge_types),
                tau=self.gumbel_temp,
                hard=True).view(old_shape)
            if return_edges:
                all_edges.append(edges)
            prev_preds, decoder_state = self.decoder(prev_preds, edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=decoder_state)
            preds.append(prev_preds)
        preds = torch.cat(preds, dim=1)
        if return_everything:
            preds = torch.cat([burn_in_predictions[:, :-1], preds], dim=1)
        if return_edges:
            return preds, torch.stack(all_edges, dim=1)
        else:
            return preds

    def copy_states(self, encoder_state, decoder_state):
        if isinstance(encoder_state, tuple) or isinstance(encoder_state, list):
            current_encoder_state = (encoder_state[0].clone(), encoder_state[1].clone())
        else:
            current_encoder_state = encoder_state.clone()
        if isinstance(decoder_state, tuple) or isinstance(decoder_state, list):
            current_decoder_state = (decoder_state[0].clone(), decoder_state[1].clone())
        else:
            current_decoder_state = decoder_state.clone()
        return current_encoder_state, current_decoder_state

    def merge_states(self, states):
        if isinstance(states[0], tuple) or isinstance(states[0], list):
            result0 = torch.cat([x[0] for x in states], dim=0)
            result1 = torch.cat([x[1] for x in states], dim=0)
            return (result0, result1)
        else:
            return torch.cat(states, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        burn_in_inputs = inputs[:, :burn_in_steps-1]
        encoder_dict = self.encoder(burn_in_inputs)
        burn_in_logits = encoder_dict['logits']
        encoder_state = encoder_dict['state']
        old_shape = burn_in_logits.shape
        burn_in_edges = nn.functional.gumbel_softmax(
            burn_in_logits.view(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=True).view(old_shape)
        burn_in_predictions, decoder_state = self.decoder(burn_in_inputs, burn_in_edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        all_timestep_preds = []
        all_timestep_edges = []
        all_edges = [] 
        for window_ind in range(burn_in_steps - 1, inputs.size(1) - 1, batch_size):
            current_batch_preds = []
            current_batch_edges = []
            encoder_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                tmp_ind = window_ind + step
                predictions = inputs[:, tmp_ind:tmp_ind+1]
                encoder_dict = self.encoder(predictions, encoder_state)
                logits = encoder_dict['logits']
                encoder_state = encoder_dict['state']
                old_shape = logits.shape
                edges = nn.functional.gumbel_softmax(
                    logits.view(-1, self.num_edge_types),
                    tau=self.gumbel_temp,
                    hard=True).view(old_shape)
                predictions, decoder_state = self.decoder(predictions, edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=decoder_state)
                current_batch_preds.append(predictions)
                if return_edges:
                    current_batch_edges.append(edges)
                tmp_encoder, tmp_decoder = self.copy_states(encoder_state, decoder_state)
                encoder_states.append(tmp_encoder)
                decoder_states.append(tmp_decoder)
            batch_encoder_state = self.encoder.merge_states(encoder_states)
            batch_decoder_state = self.merge_states(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            if return_edges:
                current_timestep_edges = [torch.cat(current_batch_edges, 0)]
                
            for step in range(prediction_steps - 1):
                encoder_dict = self.encoder(current_batch_preds, batch_encoder_state)
                logits = encoder_dict['logits']
                batch_encoder_state = encoder_dict['state']
                old_shape = logits.shape
                edges = nn.functional.gumbel_softmax(
                    logits.view(-1, self.num_edge_types),
                    tau=self.gumbel_temp,
                    hard=True).view(old_shape)
                if return_edges:
                    current_timestep_edges.append(edges)
                current_batch_preds, batch_decoder_state = self.decoder(current_batch_preds, edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=batch_decoder_state)
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.cat(current_timestep_preds, dim=1))
            if return_edges:
                all_timestep_edges.append(torch.cat(current_timestep_edges, dim=1))
        result = torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            edges = torch.cat(all_timestep_edges, dim=0)
            return result.unsqueeze(0), edges.unsqueeze(0)
        else:
            return result.unsqueeze(0)
