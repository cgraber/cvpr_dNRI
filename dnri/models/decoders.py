import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from .model_utils import encode_onehot 


class GraphRNNDecoder(nn.Module):
    def __init__(self, params):
        super(GraphRNNDecoder, self).__init__()
        self.embedder = None
        self.num_vars = num_vars =  params['num_vars']
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

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))
        if self.gpu:
            self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

    def single_step_forward(self, inputs, rel_type, hidden):
        # Inputs: [batch, num_atoms, num_dims]
        # Hidden: [batch, num_atoms, msg_out]
        # rel_type: [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        
        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                              self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                            self.msg_out_shape)
        
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
            msg = msg * rel_type[:, :, i:i+1]
            all_msgs += msg/norm

        # This step sums all of the messages per node
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1) # Average

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        pred = inputs + pred

        return pred, hidden

    def forward(self, inputs, sampled_edges, teacher_forcing=False, teacher_forcing_steps=-1, return_state=False,
                prediction_steps=-1, state=None, burn_in_masks=None):

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_feats]

        if prediction_steps > 0:
            pred_steps = prediction_steps
        else:
            pred_steps = time_steps

        if len(sampled_edges.shape) == 3:
            sampled_edges = sampled_edges.unsqueeze(1).expand(sampled_edges.size(0), pred_steps, sampled_edges.size(1), sampled_edges.size(2))
        # sampled_edges has shape:
        # [batch_size, num_time_steps, num_atoms*(num_atoms-1), num_edge_types]
        # represents the sampled edges in the graph

        # Hidden size: [batch, num_atoms, msg_out]
        if state is None:
            if inputs.is_cuda:
                hidden = torch.cuda.FloatTensor(inputs.size(0), inputs.size(2), self.msg_out_shape).fill_(0.)
            else:
                hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        else:
            hidden = state
        if teacher_forcing_steps == -1:
            teacher_forcing_steps = inputs.size(1)
        
        pred_all = []
        for step in range(0, pred_steps):
            if burn_in_masks is not None and step != 0:
                current_masks = burn_in_masks[:, step, :]
                ins = inputs[:, step, :]*current_masks + pred_all[-1]*(1 - current_masks)
            elif step == 0 or (teacher_forcing and step < teacher_forcing_steps): 
                ins = inputs[:, step, :]
            else:
                ins = pred_all[-1]
            edges = sampled_edges[:, step, :]
            pred, hidden = self.single_step_forward(ins, edges, hidden)

            pred_all.append(pred)
        preds = torch.stack(pred_all, dim=1)

        if return_state:
            return preds, hidden
        else:
            return preds