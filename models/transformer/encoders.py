from torch import nn
from models.transformer.utils import *
from models.transformer.backbone import Backbone


class MultiLevelEncoder(nn.Module):
    def __init__(self, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.padding_idx = padding_idx
    
    def forward(self, x, att, isencoder=None, memory=None, attention_weights=None):
        x1, x2, x3 = x    # [B,64*64,128], [B,32*32,256], [B,16*16,512]
        att1, att2 = att  # [B, 16, 16, h, wh//2, ww//2]: [B, 16,16, h, 4*4], [B, 16,16, h, 2*2]
        return x1, x2, x3, att1, att2


class BoostEncoder(MultiLevelEncoder):
    def __init__(self, padding_idx, d_in=512, **kwargs):
        super(BoostEncoder, self).__init__(padding_idx, **kwargs)
        self.Backbone = Backbone(d_in//8)

    def forward(self, input, isencoder=None, attention_weights=None, use_DropKey=None):
        if self.training & (use_DropKey==True):
            dropkey = True
        else:
            dropkey = False
        outs, att = self.Backbone(input, dropkey)

        x = []
        for out in outs:
            out = out.reshape(out.shape[0], out.shape[1], out.shape[2]*out.shape[3]).transpose(1,2)
            x.append(out)
        
        return super(BoostEncoder, self).forward(x, att, isencoder=isencoder, attention_weights=attention_weights)
    
