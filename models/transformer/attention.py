import numpy as np
import torch
from torch import nn
from models.containers import Module
from torch.nn import functional as F
from models.transformer.utils import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k) 10 8 50 64
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) 10 8 64 50
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) 10 8 50 64

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk) 10 8 50 50

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq,self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model) 10 50 512
        return out, att
    


class BoostEnhancedAttention(nn.Module):
    """
    Boost Enhanced cross attention
    """
    def __init__(self, d_model, d_in, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_in: Input Dimensionality of visual x
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(BoostEnhancedAttention, self).__init__()
        self.fc_v = nn.Linear(d_in, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, att12, att3, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param att12: [B, 16, 16, h, wh//2, ww//2], added pos_embed_in_enc and softmax(0~1)
        :param att3:  [B, h, nq_Word, 16*16], after softmax(0~1)
        :param values: Values (b_s, nk, d_in)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, _, nq_Word = att3.shape[:3]
        CH=CW = int(np.sqrt(att3.shape[3])) #coarse
        fh, fw = att12.shape[-2:]           #fine

        nk = values.shape[1]  # = CH*fh*CW*fw
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) 10 8 50 64

        # [B, 16, 16, h, fh, fw] -> [B, h, 16, 16, fh, fw] -> [B, h, 1, 16, 16, fh, fw]
        att = att12.permute(0,3,1,2,4,5).reshape(b_s, self.h, 1, CH,CW, fh,fw)
        att3 = att3.view(b_s, self.h, nq_Word, CH,CW, 1,1)
        # [B, h, nq_Word, 16, 16, fh, fw] -> [B, h, nq_Word, 16, fh, 16, fw] -> [B, h, nq_Word, nk]
        att = torch.mul(att, att3).permute(0,1,2,3,5,4,6).reshape(b_s, self.h, nq_Word, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, nq_Word, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model) 10 50 512
        return out




class MultiHeadAttention(Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, isenc=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            # Decoder的cross-attn是: BoostEnhancedAttention
            self.attention = attention_module(d_model=d_model, d_in=d_k, d_v=d_v, h=h)
        else:
            # Decoder的mask-attn是: ScaledDotProductAttention
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))


    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, return_att=False):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out, att = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
            
        if return_att:
            return out, att
        else:
            return out


class BoostCrossAttention(Module):
    """
    Boost Enhanced cross attention layer with Dropout and Layer Normalization.
    """
    def __init__(self, d_model, d_in, d_v, h, dropout=.1):
        super(BoostCrossAttention, self).__init__()
        self.attention = BoostEnhancedAttention(d_model=d_model, d_in=d_in, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, word_queries, att12, att3, values, attention_mask=None, attention_weights=None):
        out = self.attention(att12, att3, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(word_queries + out)
        return out