import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import dnri.models.model_utils as model_utils
from .model_utils import RefNRIMLP, encode_onehot, get_graph_info
import math


class DNRI_DynamicVars(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars, self).__init__()
        # Model Params
        self.encoder = DNRI_DynamicVars_Encoder(params)
        self.decoder = DNRI_DynamicVars_Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)

        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)

        self.burn_in_steps = params.get('train_burn_in_steps')
        self.no_prior = params.get('no_prior', False)
        self.avg_prior = params.get('avg_prior', False)
        self.learned_prior = params.get('use_learned_prior', False)
        self.anneal_teacher_forcing = params.get('anneal_teacher_forcing', False)
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.steps = 0

    def get_graph_info(self, masks):
        num_vars = masks.size(-1)
        edges = torch.ones(num_vars, device=masks.device) - torch.eye(num_vars, device=masks.device)
        tmp = torch.where(edges)
        send_edges = tmp[0]
        recv_edges = tmp[1]
        tmp_inds = torch.tensor(list(range(num_vars)), device=masks.device, dtype=torch.long).unsqueeze_(1) #TODO: should initialize as long
        edge2node_inds = (tmp_inds == recv_edges.unsqueeze(0)).nonzero()[:, 1].contiguous().view(-1, num_vars-1)
        edge_masks = masks[:, :, send_edges]*masks[:, :, recv_edges] #TODO: gotta figure this one out still
        return send_edges, recv_edges, edge2node_inds, edge_masks

    def single_step_forward(self, inputs, node_masks, graph_info, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges, node_masks, graph_info)
        return predictions, decoder_hidden, edges


    def normalize_inputs(self, inputs, node_masks):
        return self.encoder.normalize_inputs(inputs, node_masks)

    #@profile
    def calculate_loss(self, inputs, node_masks, node_inds, graph_info, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False, normalized_inputs=None):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _ = self.encoder(inputs[:, :-1], node_masks[:, :-1], node_inds, graph_info, normalized_inputs)
        if self.anneal_teacher_forcing:
            teacher_forcing_steps = math.ceil((1 - self.train_percent)*num_time_steps)
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        edge_ind = 0
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            current_node_masks = node_masks[:, step]
            node_inds = current_node_masks.nonzero()[:, -1]
            num_edges = len(node_inds)*(len(node_inds)-1)
            current_graph_info = graph_info[0][step]
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, edge_ind:edge_ind+num_edges]
            else:
                current_p_logits = prior_logits[:, edge_ind:edge_ind+num_edges]
            current_p_logits = current_p_logits.cuda(non_blocking=True)
            edge_ind += num_edges
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, current_node_masks, current_graph_info, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        target_masks = ((node_masks[:, :-1] == 1)*(node_masks[:, 1:] == 1)).float()
        loss_nll = self.nll(all_predictions, target, target_masks)
        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob.cuda(non_blocking=True), prior_logits.cuda(non_blocking=True))
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def get_prior_posterior(self, inputs, student_force=False, burn_in_steps=None):
        self.eval()
        posterior_logits = self.encoder(inputs)
        posterior_probs = torch.softmax(posterior_logits, dim=-1)
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        if student_force:
            decoder_hidden = self.decoder.get_initial_hidden(inputs)
            for step in range(burn_in_steps):
                current_inputs= inputs[:, step]
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(current_inputs, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
            for step in range(inputs.size(1) - burn_in_steps):
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(predictions, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
        else:
            for step in range(inputs.size(1)):
                current_inputs = inputs[:, step]
                prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
                all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        prior_probs = torch.softmax(logits, dim=-1)
        return prior_probs, posterior_probs

    def get_edge_probs(self, inputs):
        self.eval()
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        for step in range(inputs.size(1)):
            current_inputs = inputs[:, step]
            prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
            all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        edge_probs = torch.softmax(logits, dim=-1)
        return edge_probs

    def predict_future(self, inputs, masks, node_inds, graph_info, burn_in_masks):
        '''
        Here, we assume the following:
        * inputs contains all of the gt inputs, including for the time steps we're predicting
        * masks keeps track of the variables that are being tracked
        * burn_in_masks is set to 1 whenever we're supposed to feed in that variable's state
          for a given time step
        '''
        total_timesteps = inputs.size(1)
        prior_hidden = self.encoder.get_initial_hidden(inputs)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        predictions = inputs[:, 0]
        preds = []
        for step in range(total_timesteps-1):
            current_masks = masks[:, step]
            current_burn_in_masks = burn_in_masks[:, step].unsqueeze(-1).type(inputs.dtype)
            current_inps = inputs[:, step]
            current_node_inds = node_inds[0][step] #TODO: check what's passed in here
            current_graph_info = graph_info[0][step]
            encoder_inp = current_burn_in_masks*current_inps + (1-current_burn_in_masks)*predictions
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(encoder_inp, current_masks, current_node_inds, current_graph_info, prior_hidden)
            predictions, decoder_hidden, _ = self.single_step_forward(encoder_inp, current_masks, current_graph_info, decoder_hidden, current_edge_logits, True)
            preds.append(predictions)
        return torch.stack(preds, dim=1)

    def copy_states(self, prior_state, decoder_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        if isinstance(decoder_state, tuple) or isinstance(decoder_state, list):
            current_decoder_state = (decoder_state[0].clone(), decoder_state[1].clone())
        else:
            current_decoder_state = decoder_state.clone()
        return current_prior_state, current_decoder_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size):
        if self.fix_encoder_alignment:
            prior_logits, _, prior_hidden = self.encoder(inputs)
        else:
            prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior, tmp_decoder = self.copy_states(prior_hidden, decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
            batch_prior_hidden = self.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        return result.unsqueeze(0)

    def nll(self, preds, target, masks):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target, masks)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target, masks)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target, masks)

    def nll_gaussian(self, preds, target, masks, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))*masks.unsqueeze(-1)
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            raise NotImplementedError()
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const*masks).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1)+1e-8)
        else:
            raise NotImplementedError()


    def nll_crossent(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def nll_poisson(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.PoissonNLLLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DNRI_DynamicVars_Encoder(nn.Module):
    # Here, encoder also produces prior
    def __init__(self, params):
        super(DNRI_DynamicVars_Encoder, self).__init__()
        self.num_edges = params['num_edge_types']
        self.smooth_graph = params.get('smooth_graph', False)
        self.gpu_parallel = params.get('gpu_parallel', False)
        self.pool_edges = params.get('pool_edges', False)
        self.separate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = params['no_encoder_bn']
        dropout = params['encoder_dropout']

        hidden_size = params['encoder_hidden']
        self.rnn_hidden_size = rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.train_data_len = params.get('train_data_len', -1)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)

        self.normalize_mode = params['encoder_normalize_mode']
        if self.normalize_mode == 'normalize_inp':
            self.bn = nn.BatchNorm1d(inp_size)
        # Possible options: None, 'normalize_inp', 'normalize_all'

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
    def node2edge(self, node_embeddings, send_edges, recv_edges):
        #node_embeddings: [batch, num_nodes, embed_size]
        send_embed = node_embeddings[:, send_edges, :]
        recv_embed = node_embeddings[:, recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1) 

    def edge2node(self, edge_embeddings, edge2node_inds, num_vars):
        incoming = edge_embeddings[:, edge2node_inds[:, 0], :].clone()
        for i in range(1, edge2node_inds.size(1)):
            incoming += edge_embeddings[:, edge2node_inds[:, i]]
        return incoming/(num_vars-1)
    
    def get_initial_hidden(self, inputs):
        batch = inputs.size(0)*inputs.size(2)*(inputs.size(2)-1)
        hidden = torch.zeros(1, batch, self.rnn_hidden_size, device=inputs.device)
        cell = torch.zeros(1, batch, self.rnn_hidden_size, device=inputs.device)
        return hidden, cell

    def batch_node2edge(self, x, send_edges, recv_edges):
        send_embed = x[send_edges]
        recv_embed = x[recv_edges]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def batch_edge2node(self, x, result_shape, recv_edges):
        result = torch.zeros(result_shape, device=x.device)
        result.index_add_(0, recv_edges, x)
        return result

    def normalize_inputs(self, inputs, node_masks):
        if self.normalize_mode == 'normalize_inp':
            raise NotImplementedError
        elif self.normalize_mode == 'normalize_all':
            result = self.compute_feat_transform(inputs, node_masks)
        return result

    def compute_feat_transform(self, inputs, node_masks):
        # The following is to ensure we don't run on a sequence that's too long at once
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
            node_masks = node_masks.unsqueeze(0)
        inp_list = inputs.split(self.train_data_len, dim=1)
        node_masks_list = node_masks.split(self.train_data_len, dim=1)
        final_result = [[] for _ in range(inputs.size(0))]
        count = 0
        for inputs, node_masks in zip(inp_list, node_masks_list):
            if not self.training:
                #print("IND %d OF %d"%(count, len(node_masks_list)))
                count += 1
            flat_masks = node_masks.nonzero(as_tuple=True)
            batch_num_vars = (node_masks != 0).sum(dim=-1)
            num_vars = batch_num_vars.view(-1)
            #TODO: is it faster to put this on cpu or gpu?
            edge_info = [torch.where(torch.ones(nvar, device=num_vars.device) - torch.eye(nvar, device=num_vars.device)) for nvar in num_vars]
            offsets = torch.cat([torch.tensor([0], dtype=torch.long, device=num_vars.device), num_vars.cumsum(0)[:-1]])
            send_edges = torch.cat([l[0] + offset for l,offset in zip(edge_info, offsets)])
            recv_edges = torch.cat([l[1] + offset for l, offset in zip(edge_info, offsets)])
            
            flat_inp = inputs[flat_masks]
            tmp_batch = flat_inp.size(0)
            x = self.mlp1(flat_inp)
            x = self.batch_node2edge(x, send_edges, recv_edges)
            x = self.mlp2(x)
            x_skip = x
            result_shape = (tmp_batch, x.size(-1))
            x = self.batch_edge2node(x, result_shape, recv_edges)
            x = self.mlp3(x)
            x = self.batch_node2edge(x, send_edges, recv_edges)
            x = torch.cat([x, x_skip], dim=-1)
            x = self.mlp4(x)
            # TODO: extract into batch-wise structure
            num_edges = batch_num_vars*(batch_num_vars-1)
            num_edges = num_edges.sum(dim=-1)
            #if not self.training:
            #    x = x.cpu()
            batched_result = x.split(num_edges.tolist())
            for ind,tmp_result in enumerate(batched_result):
                final_result[ind].append(tmp_result)
        final_result = [torch.cat(tmp_result) for tmp_result in final_result]
        return final_result

    def forward(self, inputs, node_masks, all_node_inds, all_graph_info, normalized_inputs=None):
        if inputs.size(0) > 1:
            raise ValueError("Batching during forward not currently supported")
        if self.normalize_mode == 'normalize_all':
            if normalized_inputs is not None:
                x = torch.cat(normalized_inputs, dim=0)
            else:
                x = torch.cat(self.normalize_inputs(inputs, node_masks), dim=0)
        else:
            # Right now, we'll always want to do this
            raise NotImplementedError
        # Inputs is shape [batch, num_timesteps, num_vars, input_size]
        num_timesteps = node_masks.size(1)
        max_num_vars = inputs.size(2)
        max_num_edges = max_num_vars*(max_num_vars-1)
        forward_state = (torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device),
                         torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device))
        reverse_state = (torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device),
                         torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device))
        all_x = []
        all_forward_states = []
        all_reverse_states = [] 
        prior_results = []
        x_ind = 0
        for timestep in range(num_timesteps):
            current_node_masks = node_masks[:, timestep]
            node_inds = all_node_inds[0][timestep].cuda(non_blocking=True)
            if len(node_inds) <= 1:
                all_forward_states.append(torch.empty(1, 0, self.rnn_hidden_size, device=inputs.device))
                all_x.append(None)
                continue
            send_edges, recv_edges, _ = all_graph_info[0][timestep]
            send_edges, recv_edges = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            global_edge_inds = global_send_edges*(max_num_vars-1) + global_recv_edges - (global_recv_edges >= global_send_edges).long()
            current_x = x[x_ind:x_ind+len(global_send_edges)].cuda(non_blocking=True)
            x_ind += len(global_send_edges)

            old_shape = current_x.shape
            current_x = current_x.view(old_shape[-2], 1, old_shape[-1])
            current_state = (forward_state[0][:, global_edge_inds], forward_state[1][:, global_edge_inds])
            current_x, current_state = self.forward_rnn(current_x, current_state)
            tmp_state0 = forward_state[0].clone()
            tmp_state0[:, global_edge_inds] = current_state[0]
            tmp_state1 = forward_state[1].clone()
            tmp_state1[:, global_edge_inds] = current_state[1]
            all_forward_states.append(current_state[0])

        # Reverse pass
        encoder_results = []
        x_ind = x.size(0)
        for timestep in range(num_timesteps-1, -1, -1):
            current_node_masks = node_masks[:, timestep]
            node_inds = all_node_inds[0][timestep].cuda(non_blocking=True)

            if len(node_inds) <= 1:
                continue
            send_edges, recv_edges, _ = all_graph_info[0][timestep]
            send_edges, recv_edges = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            global_edge_inds = global_send_edges*(max_num_vars-1) + global_recv_edges - (global_recv_edges >= global_send_edges).long()
            current_x = x[x_ind-len(global_send_edges):x_ind].cuda(non_blocking=True)
            x_ind -= len(global_send_edges)
            old_shape = current_x.shape
            current_x = current_x.view(old_shape[-2], 1, old_shape[-1])

            current_state = (reverse_state[0][:, global_edge_inds], reverse_state[1][:, global_edge_inds])
            tmp_state0 = reverse_state[0].clone()
            tmp_state0[:, global_edge_inds] = current_state[0]
            tmp_state1 = reverse_state[1].clone()
            tmp_state1[:, global_edge_inds] = current_state[1]
            all_reverse_states.append(current_state[0])
        all_forward_states = torch.cat(all_forward_states, dim=1)
        all_reverse_states = torch.cat(all_reverse_states, dim=1).flip(1)
        all_states = torch.cat([all_forward_states, all_reverse_states], dim=-1)
        prior_result = self.prior_fc_out(all_forward_states)
        encoder_result = self.encoder_fc_out(all_states)
        return prior_result, encoder_result, forward_state
        
    def single_step_forward(self, inputs, node_masks, node_inds, all_graph_info, forward_state):
        if self.normalize_mode == 'normalize_all':
            x = self.normalize_inputs(inputs, node_masks)[0]
        else:
            raise NotImplementedError
        # Inputs is shape [batch, num_vars, input_size]
        max_num_vars = inputs.size(1)
        max_num_edges = max_num_vars*(max_num_vars-1)
        if len(node_inds) > 1:
            send_edges, recv_edges, _ = all_graph_info
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            global_edge_inds = global_send_edges*(max_num_vars-1) + global_recv_edges - (global_recv_edges >= global_send_edges).long()
            old_shape = x.shape
            x = x.view(old_shape[-2], 1, old_shape[-1])
            current_state = (forward_state[0][:, global_edge_inds], forward_state[1][:, global_edge_inds])
            x, current_state = self.forward_rnn(x, current_state)
            tmp_state0 = forward_state[0].clone()
            tmp_state0[:, global_edge_inds] = current_state[0]
            tmp_state1 = forward_state[1].clone()
            tmp_state1[:, global_edge_inds] = current_state[1]
            forward_state = (tmp_state0, tmp_state1)
            prior_result = self.prior_fc_out(current_state[0])
        else:
            prior_result = torch.empty(1, 0, self.num_edges)

        return prior_result, forward_state


class DNRI_DynamicVars_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars_Decoder, self).__init__()
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob
        
    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges, node_masks, graph_info):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, current_num_edges, num_edge_types]

        max_num_vars = inputs.size(1)
        node_inds = node_masks.nonzero()[:, -1]

        current_hidden = hidden[:, node_inds]
        current_inputs = inputs[:, node_inds]
        num_vars = current_hidden.size(1)
        
        if num_vars > 1:
            send_edges, recv_edges, edge2node_inds = graph_info
            send_edges, recv_edges, edge2node_inds = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True), edge2node_inds.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            receivers = current_hidden[:, recv_edges]
            senders = current_hidden[:, send_edges]
            # pre_msg: [batch, num_edges, 2*msg_out]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                            self.msg_out_shape, device=inputs.device)
            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2)) - 1
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2))

            # Run separate MLP for every edge type
            # NOTE: to exclude one edge type, simply offset range by 1
            for i in range(start_idx, len(self.msg_fc2)):
                msg = torch.tanh(self.msg_fc1[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2[i](msg))
                msg = msg * edges[:, :, i:i+1]
                all_msgs += msg/norm

            incoming = all_msgs[:, edge2node_inds[:, 0], :].clone()
            for i in range(1, edge2node_inds.size(1)):
                incoming += all_msgs[:, edge2node_inds[:, i], :]
            agg_msgs = incoming/(num_vars-1)
        elif num_vars == 0:
            pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
            return pred_all, hidden
        else:
            agg_msgs = torch.zeros(current_inputs.size(0), num_vars, self.msg_out_shape, device=inputs.device)

        # GRU-style gated aggregation
        inp_r = self.input_r(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_i = self.input_i(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_n = self.input_n(current_inputs).view(current_inputs.size(0), num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        current_hidden = (1 - i)*n + i*current_hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(current_hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        pred = current_inputs + pred
        hidden = hidden.clone()
        hidden[:, node_inds] = current_hidden
        pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
        pred_all[0, node_inds] = pred

        return pred_all, hidden