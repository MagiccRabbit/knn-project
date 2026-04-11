import torch
import torchaudio

class FeatureExtractor(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()

        self.n_fft=400
        self.hop_length=160
        self.n_mels=80
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.log = torchaudio.transforms.AmplitudeToDB()

    def get_features(self, wav):
        x = self.mel(wav)
        x = self.log(x)
        x = (x - x.mean()) / (x.std() + 1e-9)
        return x