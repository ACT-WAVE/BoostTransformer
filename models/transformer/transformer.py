import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('x1', None)
        self.register_state('x2', None)
        self.register_state('x3', None)
        self.register_state('att1', None)
        self.register_state('att2', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, isencoder=None, use_DropKey=False, *args):
        x1, x2, x3, att1, att2 = self.encoder(images, isencoder=isencoder, use_DropKey=use_DropKey)
        dec_output = self.decoder(seq, x1, x2, x3, att1, att2)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                
                if kwargs['freeze_backbone'] == False:
                    self.x1, self.x2, self.x3, self.att1, self.att2 = self.encoder(visual, isencoder=True)
                else:
                    self.encoder.eval()
                    with torch.no_grad():
                        self.x1, self.x2, self.x3, self.att1, self.att2 = self.encoder(visual, isencoder=True, use_DropKey=False)
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.x1, self.x2, self.x3, self.att1, self.att2)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
