import torch
import random

class AudioAugment:
    def __init__(self, noise_prob=0.3):
        self.noise_prob = noise_prob

    def __call__(self, wav):
        if random.random() < self.noise_prob:
            noise = torch.randn_like(wav) * 0.005
            wav = wav + noise

        return wav