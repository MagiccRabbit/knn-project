from transformers import WavLMModel, WavLMConfig
import torch.nn as nn
from . import ECAPA
import torch
import torch.nn.functional as F



class WavLM_ECAPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        

        for p in self.wavlm.parameters():
            p.requires_grad = False

        # unfreeze last 2 layers
        #for name, param in self.wavlm.named_parameters():

        #    if "encoder.layers.11" in name:
        #        param.requires_grad = True

        self.ecapa = ECAPA.ECAPA_TDNN(in_channels=768)

    def forward(self, x):
        with torch.no_grad():  #
            feats = self.wavlm(x).last_hidden_state  # (B, T, 768)

        emb = self.ecapa(feats)
        return emb
