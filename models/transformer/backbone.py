# -*- coding: utf-8 -*-
"""
@author: BoTiancheng_BUPT-PRIV
"""
import torch
import numpy as np
from torch import nn
from torch.nn import init


class ConvBnAct(nn.Module):
    def __init__(self, ci, co, k, s, p, d=1):  #ch_in, ch_out, kernel, stride, padding, dilation
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(co, eps=1e-5, momentum=0.1)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, X):
        return self.act(self.bn(self.conv(X)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ConvBn(nn.Module):
    def __init__(self, ci, co, k=1, s=1, p=0, d=1, g=1):
        # ch_in, ch_out, kernel, stride, padding, dilation, groups
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, dilation=d, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(co, eps=1e-5, momentum=0.1)

    def forward(self, X):
        return self.bn(self.conv(X))
    
    def fuseforward(self, x):
        return self.conv(x)


class MP(nn.Module):
    def __init__(self, ci):  #输入输出通道数均为ci
        super(MP, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv = ConvBn(ci, ci, k=4, s=2, p=1, g=ci)

    def forward(self, x):
        x_m = self.mp(x)
        x_c = self.cv(x)
        return torch.cat([x_m, x_c], 1)


class DilationBranch(nn.Module):
    def __init__(self, ci, co):
        super(DilationBranch,self).__init__()
        self.conv1_a = ConvBnAct(ci,co,3,1,1,1)
        self.conv2_a = ConvBnAct(ci,co,3,1,2,2)
        self.conv1_b = ConvBnAct(ci,co,3,1,2,2)
        self.conv2_b = ConvBnAct(ci,co,3,1,3,3)
        self.conv1_c = ConvBnAct(ci,co,3,1,3,3)
        self.conv2_c = ConvBnAct(ci,co,3,1,4,4)
        
    def forward(self,X):
        DBa = self.conv2_a(self.conv1_a(X))
        DBb = self.conv2_b(self.conv1_b(X))
        DBc = self.conv2_c(self.conv1_c(X))
        return torch.cat([DBa, DBb, DBc], 1)

class Stem(nn.Module):
    def __init__(self, d_model):
        super(Stem,self).__init__()
        self.DownSample = nn.Sequential(
                          ConvBnAct( 3,16,5,2,2),       #256,3 -> 128,16
                          ConvBnAct(16,32,3,2,1))       #128,16 -> 64,32
        self.DilationBranch = DilationBranch(32,32)     # 64,32 -> 64,96
        self.InputProj = ConvBnAct(96,2*d_model,3,1,1)  # 64,96 -> 64,128
        self.OutProj = ConvBnAct(2*d_model, 2*d_model, k=3,s=1,p=1)
        
    def forward(self, x):
        x = self.InputProj(self.DilationBranch(self.DownSample(x)))
        x = self.OutProj(x)
        return x


class MHSA(nn.Module):
    def __init__(self, mode, d_model, qk_dim, h, resolution, 
                        gather_arg1=None, gather_arg2=None):
        '''
        q和k的维度相同，比v小；k和v的patch数目相同，可以跟q的patch数目不一致
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys:16/h
        :param d_v: Dimensionality of values:64/h
        :param h: Number of heads
        :param resolution: size of input feature map
        '''
        super(MHSA, self).__init__()
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.d_k = qk_dim//h
        self.d_v = d_model//h
        self.h = h
        self.resolution = resolution
        self.Wqk = ConvBn(d_model, 2*qk_dim)
        self.Wv  = ConvBn(d_model, d_model)
        self.fc_o = nn.Linear(h*self.d_v, d_model)
        
        self.pos_embedding, idxs = self.create_pos_embedding_and_idxs(h, resolution, resolution)
        self.register_buffer("relative_pos_index", idxs)
        self.mode = mode
        if (gather_arg1 != None) or (gather_arg2 != None): 
            self.Index = self.gather_indices(gather_arg1, gather_arg2)
        self.init_weights()
    
    
    def gather_indices(self, arg1, arg2):
        '''
        arg1 : Bool - Decoder or not
        arg2 : Int - special rate
        '''
        if arg1 == False:
            CH = CW = self.resolution*arg2
            nh = nw = 2*arg2
        else:
            CH = CW = self.resolution*2  #coarse attn的大小（ca=32*32；ca=16*16）
            nh = nw = 8   #V中有多少个窗（窗长16，V=128*128；窗长8，V=64*64）

        cwh, cww = CH//nh, CW//nw
        num_windows = nh*nw
        ca_index = torch.ones((1,self.h,CH*CW,num_windows), dtype=torch.long)#, device=ca.device)
        for i in range(CH):
            si, ti = i%cwh*cww, i*CW
            for j in range(CW):
                ca_index[:,:,ti+j] = torch.tensor([si+j%cww + cwh*cww*k for k in range(num_windows)])
        return ca_index


    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    #2d_relative_bias_trainable_embeddings
    def create_pos_embedding_and_idxs(self, n_head, height, width): 
        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width))) #[2,h,w]
        coords_flatten = torch.flatten(coords, 1)  #[2,h*w]
        relative_coords_bias = coords_flatten[:,:,None] - coords_flatten[:,None,:] #[2,h*w,h*w]
        relative_coords_bias[0,:,:] += height-1
        relative_coords_bias[1,:,:] += width-1
        relative_coords_bias[0,:,:] *= 2*width-1  # A:2d->B:1d, B[i*cols+j]=A[i,j]
        
        relative_position_index = relative_coords_bias.sum(0) #[height*width,height*width]
        position_embedding = nn.Parameter(torch.zeros(n_head, (2*height-1)*(2*width-1)))
        return position_embedding, relative_position_index


    def forward_vanilla(self, x, mask_ratio, use_DropKey):
        '''
        :param x: feature map [B, C, H, W]
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        '''
        b_s, C, H, W = x.shape
        q, k = self.Wqk(x).split([self.qk_dim, self.qk_dim], dim=1) #B,64,32,32 -> B,32,32,32 -> [B,16,32,32]*2
        v = self.Wv(x)  #B,64,32,32 -> B,64,32,32
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        nq, nk = q.shape[2], k.shape[2]
        
        q = q.view(b_s, self.h, self.d_k, nq).transpose(2, 3) #(b_s, h, nq, d_k)
        k = k.view(b_s, self.h, self.d_k, nk)                 #(b_s, h, d_k, nk)
        v = v.view(b_s, self.h, self.d_v, nk).transpose(2, 3) #(b_s, h, nk, d_v)
        bias_embedding = self.pos_embedding[:, self.relative_pos_index].unsqueeze(0)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq*nk)
        if use_DropKey == True:
            m_r = torch.ones_like(att) * mask_ratio
            att = att + torch.bernoulli(m_r) * -1e12
        
        att = att + bias_embedding
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, nq, self.h*self.d_v)
        
        #(b_s,nq,h*d_v)->(B,nq,C)->(B,C,nq)->[B,C,H,W]
        out = self.fc_o(out).permute(0,2,1).view(b_s,C,H,W) 
        return out, att
    
    
    def forward_expand(self, x, mask_ratio, use_DropKey, ca, v):
        '''
        :param x: qk-fmap in window-format: [B*nh*nw C wh ww]
        :param ca: coarse attention input [B heads coarse_H*coarse_W coarse_H*coarse_W]
        :param v: origin_V-fmap in normal-format: [B C VH VW]
        '''
        b_s, C, wh, ww = x.shape
        B, _, VH, VW = v.shape
        nh = torch.div(VH, wh, rounding_mode='trunc')
        nw = torch.div(VW, ww, rounding_mode='trunc')
        CH=CW = int(np.sqrt(ca.shape[2]))
        
        scaleH = torch.div(VH, CH, rounding_mode='trunc').to(ca.device)
        scaleW = torch.div(VW, CW, rounding_mode='trunc').to(ca.device)
        cwh = torch.div(CH, nh, rounding_mode='trunc')
        cww = torch.div(CW, nw, rounding_mode='trunc')
        
        q, k = self.Wqk(x).split([self.qk_dim, self.qk_dim], dim=1) #B,64,32,32 -> B,32,32,32 -> [B,16,32,32]*2
        v = self.Wv(v)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        nq, nk, nv = q.shape[2], k.shape[2], v.shape[2]  #wh*ww wh*ww VH*VW
        
        q = q.view(b_s, self.h, self.d_k, nq).transpose(2, 3) #(b_s, heads, nq, d_k)
        k = k.view(b_s, self.h, self.d_k, nk)                 #(b_s, heads, d_k, nk)
        v = v.view(B, self.h, self.d_v, nv).transpose(2, 3)   #(B, heads, VH*VW, d_v)
        bias_embedding = self.pos_embedding[:, self.relative_pos_index].unsqueeze(0)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, heads, nq*nk)
        if use_DropKey == True:
            m_r = torch.ones_like(att) * mask_ratio
            att = att + torch.bernoulli(m_r) * -1e12
        att = att + bias_embedding
        att = torch.softmax(att, -1)

        #att窗口形式转常规形式，这样fa和ca都是常规形式
        #把其他窗口点的权重拼给第一个窗口的点，使第一个窗口点的权重有常规形式排列
        att = att.view(B,nh,nw,self.h,wh*ww,wh,ww).permute(0,3,4,1,5,2,6).reshape(B,self.h,wh,ww,VH*VW)
        
        #隔点取值
        ca_e = ca.clone()
        ca_e = ca_e.view(B,self.h, CH*CW, nh,cwh,nw,cww).transpose(4,5).reshape(B,self.h,CH*CW,CH*CW)
        ca_index = self.Index.to(ca.device).repeat(B,1,1,1)
        ca_e = torch.gather(ca_e, 3, ca_index)
        ca_e = ca_e.view(B, self.h, CH,1, CW,1, nh, nw)
        ca_e = ca_e.repeat(1, 1, 1, scaleH, 1, scaleW, 1, 1)
        ca_e = ca_e.view(B, self.h, VH*VW, nh, nw)
        
        #广播机制矩阵相乘：att(B,heads, wh,ww, VH*VW)， ca(B,heads, VH*VW, nh,nw)
        att = att.view(B,self.h,  1, wh,  1, ww, nh, wh, nw, ww)
        ca_e=ca_e.view(B,self.h, nh, wh, nw, ww, nh,  1, nw,  1)
        att = torch.mul(att, ca_e)
        att = att.view(B, self.h, VH*VW, VH*VW)  #att与v都是常规形式可以直接乘
        
        #[B,heads,VH*VW,VH*VW] * [B,heads,VH*VW,d_v] = [B,heads,VH*VW,d_v]
        out = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, nv, self.h*self.d_v)
        #(B,VH*VW,h*d_v) -> (B,VH*VW,C) -> (B,VH,VW,C) 常规形式排列
        out = self.fc_o(out).view(B, VH, VW, C)
        return out, att
    
    
    def forward_compact(self, x, mask_ratio, use_DropKey, ca, origin_size):
        '''
        :param x: qkv-fmap in window-format: [B*nh*nw C wh ww]
        :param ca: coarse attention input [B heads coarse_H*coarse_W coarse_H*coarse_W]
        :param origin_size: list[B, VH, VW, VRAM-divisor]
        '''
        b_s, C, wh, ww = x.shape
        B, VH, VW, divisor = origin_size
        nh = torch.div(VH, wh, rounding_mode='trunc')
        nw = torch.div(VW, ww, rounding_mode='trunc')
        
        CH=CW = int(np.sqrt(ca.shape[2]))
        scaleH = torch.div(VH, CH, rounding_mode='trunc').to(ca.device)
        scaleW = torch.div(VW, CW, rounding_mode='trunc').to(ca.device)
        cwh = torch.div(CH, nh, rounding_mode='trunc')
        cww = torch.div(CW, nw, rounding_mode='trunc')
        
        q, k = self.Wqk(x).split([self.qk_dim, self.qk_dim], dim=1) #B,64,32,32 -> B,32,32,32 -> [B,16,32,32]*2
        v = self.Wv(x)  #[B VH VW C] -> [B C VH VW]
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        nq, nk = q.shape[2], k.shape[2] #wh*ww wh*ww
        
        q = q.view(b_s, self.h, self.d_k, nq).transpose(2, 3) #(b_s, heads, nq, d_k)
        k = k.view(b_s, self.h, self.d_k, nk)                 #(b_s, heads, d_k, nk)
        v = v.view(b_s, self.h, self.d_v, nk).transpose(2, 3) #(b_s, heads, nk, d_v)
        bias_embedding = self.pos_embedding[:, self.relative_pos_index].unsqueeze(0)
        
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, heads, nq*nk)
        if use_DropKey == True:
            m_r = torch.ones_like(att) * mask_ratio
            att = att + torch.bernoulli(m_r) * -1e12
        att = att + bias_embedding
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).view(B, nh*nw, self.h, wh, ww, self.d_v)
        out = out.permute(0, 2, 5, 3, 4, 1).reshape(B, self.h, self.d_v, wh*ww, nh*nw)
        
        #隔点取值        
        ca_e = ca.clone()
        ca_e = ca_e.view(B,self.h, CH*CW, nh,cwh,nw,cww).transpose(4,5).reshape(B,self.h,CH*CW,CH*CW)
        ca_index = self.Index.to(ca.device).repeat(B,1,1,1)
        ca_e = torch.gather(ca_e, 3, ca_index)
        ca_e = ca_e.view(B, self.h, CH,1, CW,1, nh, nw)
        ca_e = ca_e.repeat(1, 1, 1, scaleH, 1, scaleW, 1, 1)
        ca_e = ca_e.view(B, self.h, VH*VW, nh, nw)
        
        #广播+分块机制矩阵相乘：out(B,h, d_v, wh*ww, nh*nw)， ca_e(B, h, VH*VW, nh*nw)
        out = out.view(B, self.h, self.d_v,  1, wh,  1, ww, nh*nw)
        ca_e=ca_e.view(B, self.h,        1, nh, wh, nw, ww, nh*nw)
        
        outC = torch.zeros((B,self.h, self.d_v, nh,wh, nw,ww), device=ca.device)
        chunk_size = self.d_v//divisor
        for i in range(0, self.d_v, chunk_size):
            out_chunk = out[:,:, i:i+chunk_size, :,:,:,:,:]
            outC[:,:, i:i+chunk_size, :,:,:,:] = torch.mul(out_chunk, ca_e).sum(-1)
        out = outC.view(B, self.h*self.d_v, VH*VW).transpose(1,2)

        out = self.fc_o(out).view(B, VH, VW, C)
        return out
    

    def forward_compact_down(self, x, mask_ratio, use_DropKey, ca, origin_size):
        '''
        :param x: qkv-fmap in window-format: [B*nh*nw C wh ww]
        :param ca: coarse attention input [B heads coarse_H*coarse_W coarse_H*coarse_W]
        :param origin_size: list[B, VH, VW, VRAM-divisor]
        '''
        b_s, C, wh, ww = x.shape
        B, VH, VW, divisor = origin_size
        nh = torch.div(VH, wh, rounding_mode='trunc')
        nw = torch.div(VW, ww, rounding_mode='trunc')
        
        CH=CW = int(np.sqrt(ca.shape[2]))
        scaleH = torch.div(VH, CH, rounding_mode='trunc').to(ca.device)
        scaleW = torch.div(VW, CW, rounding_mode='trunc').to(ca.device)
        cwh = torch.div(CH, nh, rounding_mode='trunc')
        cww = torch.div(CW, nw, rounding_mode='trunc')
        
        q, k = self.Wqk(x).split([self.qk_dim, self.qk_dim], dim=1) #B,64,32,32 -> B,32,32,32 -> [B,16,32,32]*2
        v = self.Wv(x)  #[B VH VW C] -> [B C VH VW]
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        nq, nk = q.shape[2], k.shape[2] #wh*ww wh*ww
        
        q = q.view(b_s, self.h, self.d_k, nq).transpose(2, 3) #(b_s, heads, nq, d_k)
        k = k.view(b_s, self.h, self.d_k, nk)                 #(b_s, heads, d_k, nk)
        v = v.view(b_s, self.h, self.d_v, nk).transpose(2, 3) #(b_s, heads, nk, d_v)
        bias_embedding = self.pos_embedding[:, self.relative_pos_index].unsqueeze(0)
        
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, heads, nq*nk)
        if use_DropKey == True:
            m_r = torch.ones_like(att) * mask_ratio
            att = att + torch.bernoulli(m_r) * -1e12

        att = att + bias_embedding
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).view(B, nh*nw, self.h, wh, ww, self.d_v)
        out = out.permute(0, 2, 5, 3, 4, 1).reshape(B, self.h, self.d_v, wh*ww, nh*nw)
        
        #隔点取值        
        ca_e = ca.clone()
        ca_e = ca_e.view(B,self.h, CH*CW, nh,cwh,nw,cww).transpose(4,5).reshape(B,self.h,CH*CW,CH*CW)
        ca_index = self.Index.to(ca.device).repeat(B,1,1,1)
        ca_e = torch.gather(ca_e, 3, ca_index)
        ca_e = ca_e.view(B, self.h, CH,1, CW,1, nh, nw)
        ca_e = ca_e.repeat(1, 1, 1, scaleH, 1, scaleW, 1, 1)
        ca_e = ca_e.view(B, self.h, VH*VW, nh, nw)
        
        #广播+分块机制矩阵相乘：out(B,h, d_v, wh*ww, nh*nw)， ca_e(B, h, VH*VW, nh*nw)
        out = out.view(B, self.h, self.d_v,  1, wh,  1, ww, nh*nw)
        ca_e=ca_e.view(B, self.h,        1, nh, wh, nw, ww, nh*nw)
        
        outC = torch.zeros((B,self.h, self.d_v, nh,wh, nw,ww), device=ca.device)
        chunk_size = self.d_v//divisor
        for i in range(0, self.d_v, chunk_size):
            out_chunk = out[:,:, i:i+chunk_size, :,:,:,:,:]
            outC[:,:, i:i+chunk_size, :,:,:,:] = torch.mul(out_chunk, ca_e).sum(-1)
        out = outC.view(B, self.h*self.d_v, VH*VW).transpose(1,2)
        out = self.fc_o(out).view(B, VH, VW, C)

        att_out = att #add pos_embed and softmax(0~1)
        att_out = att_out.view(B, nh,nw, self.h, wh*ww, wh*ww).mean(dim=-2) #保留key维作为每个位置“被关注的程度”
        # 32: [B, 8,8, h, 4*4, 4*4] --mean-> [B, 8,8, h, 4*4] --view+weight-> [B, 16,16, h, 2*2] --reshape-> [B, h, 32*32]
        # 64: [B, 8,8, h, 8*8, 8*8] --mean-> [B, 8,8, h, 8*8] --view+weight-> [B, 16,16, h, 4*4] --reshape-> [B, h, 64*64]
        att_out = att_out.view(B, nh,nw, self.h, 2,wh//2, 2,ww//2).permute(0,1,4,2,6,3,5,7).reshape(
                               B, nh*2, nw*2, self.h, wh//2, ww//2)
        #print(att_out.shape)
        return out, att_out #处理后的16*16个窗口形式的att，用于MeshedDecoder

    
    def forward(self, x, drop_ratio, use_DropKey, coarse_attin=None, divisor=None):
        if self.mode == 'vanilla':
            x, att_out = self.forward_vanilla(x, drop_ratio, use_DropKey)
            return x, att_out
        else:
            B, C, H, W = x.shape
            pad_b = (self.resolution - H % self.resolution) % self.resolution
            pad_r = (self.resolution - W % self.resolution) % self.resolution
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = torch.nn.functional.pad(x, (0,pad_r, 0,pad_b))
            pH, pW = H + pad_b, W + pad_r
            nH = torch.div(pH, self.resolution, rounding_mode='trunc')
            nW = torch.div(pW, self.resolution, rounding_mode='trunc')
            origin_V = x
            # window partition, BCHW -> BC(nHh)(nWw) -> BnHnWChw -> (BnHnW)Chw
            x = x.view(B, C, nH,self.resolution, nW,self.resolution).permute(0,2,4,1,3,5).reshape(
                       B*nH*nW, C, self.resolution, self.resolution)
            
            if self.mode == 'expand':
                x, att_out = self.forward_expand(x, drop_ratio, use_DropKey, coarse_attin, origin_V)
                if padding:
                    x = x[:, :H, :W].contiguous()
                x = x.permute(0, 3, 1, 2).contiguous()
                return x, att_out
            
            else:
                if self.mode == 'compact':
                    x = self.forward_compact(x, drop_ratio, use_DropKey, coarse_attin, [B,H,W,divisor])
                    if padding:
                        x = x[:, :H, :W].contiguous()
                    x = x.permute(0, 3, 1, 2).contiguous()
                    return x
                elif self.mode == 'compact_down':
                    x, att_out = self.forward_compact_down(x, drop_ratio, use_DropKey, coarse_attin, [B,H,W,divisor])
                    if padding:
                        x = x[:, :H, :W].contiguous()
                    x = x.permute(0, 3, 1, 2).contiguous()
                    return x, att_out


    
class FFN(nn.Module):
    '''
    CNN-based Feed Forward Network
    '''
    def __init__(self, in_dim, hidden_dim, drop=0.05):
        super(FFN, self).__init__()
        self.conv1 = ConvBnAct(in_dim, hidden_dim, 1,1,0)
        self.conv2 = ConvBn(hidden_dim, in_dim)
        self.dropout = nn.Dropout(drop)
    
    def forward(self,X):
        return self.dropout(self.conv2(self.conv1(X)))
        
    
class BoostBlock(nn.Module):
    def __init__(self, d_model, h, window_size, down=False):
        super(BoostBlock,self).__init__()
        self.down = down
        qk_dim = d_model//4
        
        #8*8分支“4”，即4倍下采样低分辨率检测大目标分支
        self.down_patch_4 = ConvBn(d_model,d_model,k=6,s=4,p=1,g=d_model) #overlap patch embedding
        self.attn4 = MHSA('vanilla', d_model, qk_dim, h, window_size*2)
        self.norm_4a = nn.BatchNorm2d(d_model)
        self.FFN4 = FFN(d_model, 2*d_model)
        self.norm_4b = nn.BatchNorm2d(d_model)
        #亚像素卷积
        self.upsample4 = nn.Sequential(
            nn.Conv2d(d_model, d_model*4, 1,1,0), nn.PixelShuffle(upscale_factor=2))
        
        #16*16分支“2”，即2倍下采样中分辨率检测中目标分支
        self.down_patch_2 = ConvBn(d_model,d_model,k=4,s=2,p=1,g=d_model) #overlap patch embedding
        self.attn2 = MHSA('expand', d_model, qk_dim, h, window_size, False,2)
        self.norm_2a = nn.BatchNorm2d(d_model)
        self.FFN2 = FFN(d_model, 2*d_model)
        self.norm_2b = nn.BatchNorm2d(d_model)
        #双线性插值
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        #32*32分支“1”，即1倍下采样高分辨率检测小目标分支
        attn1_mode = 'compact' if not down else 'compact_down' #准备输出注意力权重矩阵给decoder
        self.attn1 = MHSA(attn1_mode, d_model, qk_dim, h, window_size, False,4)
        self.norm_1a = nn.BatchNorm2d(d_model)
        self.FFN1 = FFN(d_model, 2*d_model)
        self.norm_1b = nn.BatchNorm2d(d_model)
        self.OutProj = ConvBnAct(d_model,d_model,3,1,1)
        
        if down:
            self.MP = MP(d_model)
            self.double_channel = ConvBnAct(d_model, 2*d_model, k=1,s=1,p=0)
            self.OutProj_down = ConvBnAct(2*d_model, 2*d_model, k=3,s=1,p=1)
        
        
    def forward(self, fmap, drop_ratio, use_DropKey):
        #8*8分支“4”，即4倍下采样低分辨率检测大目标分支
        fmap4 = self.down_patch_4(fmap)   #B,C,32,32-> B,C,8,8
        out4, att_out = self.attn4(fmap4, drop_ratio/4, use_DropKey)
        out4 = self.norm_4a(out4 + fmap4)
        out4 = self.norm_4b(out4 + self.FFN4(out4))
        out4 = self.upsample4(out4)  #上采样 B,C,8,8 -> B,C,16,16
        
        #16*16分支“2”，即2倍下采样中分辨率检测中目标分支
        fmap2 = self.down_patch_2(fmap)   #B,C,32,32-> B,C,16,16
        out2, att_out = self.attn2(fmap2, drop_ratio/2, use_DropKey, att_out)
        out2 = self.norm_2a(out2 + fmap2)
        out2 = self.norm_2b(out2 + self.FFN2(out2))
        out2_d = out2 + out4
        out2 = self.upsample2(out2_d) #上采样 B,C,16,16 -> B,C,32,32
        
        #32*32分支“1”，即1倍下采样高分辨率检测小目标分支
        if not self.down:
            out1 = self.attn1(fmap, drop_ratio, use_DropKey, att_out, 4)
            out1 = self.norm_1a(out1 + fmap)
            out1 = self.norm_1b(out1 + self.FFN1(out1))
            out = self.OutProj(out1 + out2)
            return out #B,C,32,32
        
        else:
            out1, att_out = self.attn1(fmap, drop_ratio, use_DropKey, att_out, 4)
            out1 = self.norm_1a(out1 + fmap)
            out1 = self.norm_1b(out1 + self.FFN1(out1))
            out = self.OutProj(out1 + out2)

            out1_d = self.MP(out1)
            out2_d = self.double_channel(out2_d)
            out_d = self.OutProj_down(out1_d + out2_d)
            return out, att_out, out_d
        
        
class BoostBlock_E(nn.Module):
    def __init__(self, d_model, h, window_size):
        super(BoostBlock_E,self).__init__()
        qk_dim = d_model//4
        #8*8分支“2”
        self.down_patch_2 = ConvBn(d_model,d_model,k=4,s=2,p=1,g=d_model) #overlap patch embedding
        self.attn2 = MHSA('vanilla', d_model, qk_dim, h, window_size*2)
        self.norm_2a = nn.BatchNorm2d(d_model)
        self.FFN2 = FFN(d_model, 2*d_model)
        self.norm_2b = nn.BatchNorm2d(d_model)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        #16*16分支“1”
        self.attn1 = MHSA('compact', d_model, qk_dim, h, window_size, False,2)
        self.norm_1a = nn.BatchNorm2d(d_model)
        self.FFN1 = FFN(d_model, 2*d_model)
        self.norm_1b = nn.BatchNorm2d(d_model)
        self.OutProj = ConvBnAct(d_model,d_model,3,1,1)
        
        
    def forward(self, fmap, drop_ratio, use_DropKey):
        #8*8分支“2”
        fmap2 = self.down_patch_2(fmap)   #B,C,16,16-> B,C,8,8
        out2, att_out = self.attn2(fmap2, drop_ratio/2, use_DropKey)
        out2 = self.norm_2a(out2 + fmap2)
        out2 = self.norm_2b(out2 + self.FFN2(out2))
        out2 = self.upsample2(out2) #上采样 B,C,8,8 -> B,C,16,16
        
        #16*16分支“1”
        out1 = self.attn1(fmap, drop_ratio, use_DropKey, att_out, 4)
        out1 = self.norm_1a(out1 + fmap)
        out1 = self.norm_1b(out1 + self.FFN1(out1))
        out = self.OutProj(out1 + out2)
        return out #B,C,16,16
        
        

class Backbone(nn.Module):
    def __init__(self, nC):
        super().__init__()
        self.Stem = Stem(nC)                  # 256*256/512*512 3 -> 64*64 128
        self.BT1D = BoostBlock(nC*2, 8, 8, down=True) # 64*64 128 -> 32*32 256
        
        self.BT2A = BoostBlock(nC*4, 4, 4)            # 32*32 256 -> 32*32 256
        self.BT2B = BoostBlock(nC*4, 4, 4)            # 32*32 256 -> 32*32 256
        self.BT2D = BoostBlock(nC*4, 8, 4, down=True) # 32*32 256 -> 16*16 512
        
        self.BT3A = BoostBlock_E(nC*8, 8, 4)          # 16*16 512 -> 16*16 512
        self.BT3B = BoostBlock_E(nC*8, 8, 4)          # 16*16 512 -> 16*16 512
        self.BT3C = BoostBlock_E(nC*8, 8, 4)          # 16*16 512 -> 16*16 512
    
    
    def forward(self, x, dropkey):
        x1 = self.Stem(x)                          # 64*64 128
        x1,att1, x2 = self.BT1D(x1, 0.16, dropkey) # 64*64 128, 32*32 256
        
        x2 = self.BT2A(x2, 0.08, dropkey)          # 32*32 256
        x2 = self.BT2B(x2, 0.08, dropkey)          # 32*32 256
        x2,att2, x3 = self.BT2D(x2, 0.08, dropkey) # 32*32 256, 16*16 512
        
        x3 = self.BT3A(x3, 0.04, dropkey)          # 16*16 512
        x3 = self.BT3B(x3, 0.04, dropkey)          # 16*16 512
        x3 = self.BT3C(x3, 0.04, dropkey)          # 16*16 512

        return [x1, x2, x3], [att1, att2]
