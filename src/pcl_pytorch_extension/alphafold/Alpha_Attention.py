import Alpha_Attention_cpp

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
# from pcl_pytorch_extension.utils.blocked_layout import (
#     BlockedParameter,
#     BlockedModule,
#     BlockedTensor,
#     get_blocking_signature,
# )
from pcl_pytorch_extension._C import _alpha_attention as Alpha_Attention_cpp
import time
from contextlib import contextmanager

class AlphaAttentionFunction(Function):
  @staticmethod
  def forward(ctx, q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, \
    gating_w, gating_b, output_w, output_b, key_dim, value_dim):
    result = Alpha_Attention_cpp.forward(q_data, m_data, bias, nonbatched_bias, query_w, key_w, value_w, \
      gating_w, gating_b, output_w, output_b, key_dim, value_dim )
    return result

def GatingAttentionOpti_forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """

    output = AlphaAttentionFunction.apply(q_data, m_data, bias, nonbatched_bias, 
                 self.query_w, self.key_w, self.value_w,
                 self.gating_w, self.gating_b, self.output_w, self.output_b, self.key_dim, self.value_dim )
    return output

class GatingAttentionOpti(nn.Module):
  """Multihead attention w/ Gating"""

  # def __init__(self, config, global_config, a_dim, m_dim, output_dim):
  def __init__(self, num_head, a_dim, m_dim, output_dim):
    super().__init__()
    # self.config = config
    # self.global_config = global_config
    self.output_dim = output_dim
    # k,v dim
    # self.key_dim = self.config.get('key_dim', int(a_dim))
    # self.value_dim = self.config.get('value_dim', int(m_dim))
    # self.num_head = self.config['num_head']
    self.key_dim = int(a_dim)
    self.value_dim = int(m_dim)
    self.num_head = num_head
    assert self.key_dim % self.num_head == 0
    assert self.value_dim % self.num_head == 0
    self.key_dim = self.key_dim // self.num_head
    self.value_dim = self.value_dim // self.num_head
    # q,k,v weights
    self.query_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.key_dim),requires_grad=False)
    self.key_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.key_dim),requires_grad=False)
    self.value_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.value_dim),requires_grad=False)
    self.gating_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.value_dim),requires_grad=False)
    self.gating_b = nn.Parameter(torch.Tensor(self.num_head,self.value_dim),requires_grad=False)
    self.output_w = nn.Parameter(torch.Tensor(self.num_head,self.value_dim, self.output_dim),requires_grad=False)
    self.output_b = nn.Parameter(torch.Tensor(self.output_dim),requires_grad=False)
    # softmax & act fn
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()

  
  @torch.jit.ignore
  def read_time(self) -> float:
    return time.time()


  def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """

    output = AlphaAttentionFunction.apply(q_data, m_data, bias, nonbatched_bias, 
                 self.query_w, self.key_w, self.value_w,
                 self.gating_w, self.gating_b, self.output_w, self.output_b, self.key_dim, self.value_dim )

    # # get query, key, value
    # q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_dim**(-0.5)
    # k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w)
    # v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w)

    # logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias

    # if nonbatched_bias.shape[0] > 0:
    #   logits += torch.unsqueeze(nonbatched_bias, dim=0)
    # weights = self.softmax(logits)

    # weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)

    # gate_values = torch.einsum('bqc,chv->bqhv', q_data,self.gating_w) + self.gating_b
    # gate_values = self.sigmoid(gate_values)
    # weighted_avg *= gate_values

    # output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    return output


