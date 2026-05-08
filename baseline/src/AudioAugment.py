import torch
import torch.nn as nn
from pathlib import Path

from torch_audiomentations import (
    Compose,
    AddBackgroundNoise,
    ApplyImpulseResponse,
    PitchShift,
    Gain,
    PolarityInversion,
)

class RescaledImpulseResponse(nn.Module):
    def __init__(self, ir_paths, p=1.0, sample_rate=16000):
        super().__init__()
        self.ir_transform = ApplyImpulseResponse(
            ir_paths=ir_paths, p=p, output_type="tensor"
        )
        self.sample_rate = sample_rate

    def forward(self, samples, sample_rate=None):
        # 1. Capture the 'Dry' RMS (original volume)
        # Using dim=-1 to calculate RMS per channel/track in the batch
        input_rms = torch.sqrt(torch.mean(samples**2, dim=-1, keepdim=True))

        # 2. Apply the reverb
        # We pass self.sample_rate if the pipeline doesn't provide one
        sr = sample_rate if sample_rate is not None else self.sample_rate
        augmented = self.ir_transform(samples, sample_rate=sr)

        # 3. Capture the 'Wet' RMS (quiet reverb volume)
        output_rms = torch.sqrt(torch.mean(augmented**2, dim=-1, keepdim=True))

        # 4. Match levels: Scale 'Wet' back to 'Dry'
        # Add a tiny epsilon to avoid division by zero
        return augmented * (input_rms / (output_rms + 1e-7))


class AudioAugment:
    def __init__(
        self,
        noise_root_dir: Path,
        rir_root_dir: Path,
        sample_rate=16000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.augment_wav = Compose(
            [
                # Apply reverb
                RescaledImpulseResponse(
                    ir_paths=str(rir_root_dir), p=0.4
                ),
                # Add background noise
                AddBackgroundNoise(
                    background_paths=str(noise_root_dir),
                    min_snr_in_db=5.0,
                    max_snr_in_db=20.0,
                    p=0.5,
                    output_type="tensor",
                ),
                # Vocal tract variation
                PitchShift(
                    min_transpose_semitones=-4,
                    max_transpose_semitones=4,
                    p=0.2,
                    sample_rate=self.sample_rate,
                    output_type="tensor",
                ),
                # Microphone distance simulation
                Gain(
                    min_gain_in_db=-7.0, max_gain_in_db=7.0, p=0.3, output_type="tensor"
                ),
                # Invert waveform
                PolarityInversion(p=0.5, output_type="tensor"),
            ],
            output_type="tensor",
        ).to(self.device)

    def __call__(self, wav_batch):
        return self.augment_wav(samples=wav_batch, sample_rate=self.sample_rate)
