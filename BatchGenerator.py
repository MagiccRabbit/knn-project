import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import librosa
import imageio_ffmpeg as ffmpeg
import subprocess
from pathlib import Path

class BatchGenerator(Dataset):
    def __init__(self, root_dir, sample_rate=16000, segment_len=3.0, speakers_num = 32, segments_num = 6):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.speakers_num = speakers_num
        self.segments_num = segments_num

        dataset_path = Path(self.root_dir)
        self.speaker_paths = [spk for spk in dataset_path.iterdir() if spk.is_dir()]
        self.speaker_ids = {spk.name:index for index, spk in enumerate(self.speaker_paths)}
        print(self.speaker_ids)

    def _load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sample_rate,  mono=True)
        wav = torch.from_numpy(wav)
        wav = wav.squeeze(0)
        return wav

    def _get_random_segment(self, path):
        wav = self._load_wav(path)
        seg_len = int(self.sample_rate * self.segment_len)

        if len(wav) < seg_len:
            # padding
            pad = seg_len - len(wav)
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            start = random.randint(0, len(wav) - seg_len)
            wav = wav[start:start + seg_len]

        return wav
    
    def generate_random_speaker_balanced_batch(self):
        
        selected_spks = random.sample(self.speaker_paths, self.speakers_num)

        batch = []
        labels = []

        for spk in selected_spks:
            wav_paths = [f for f in spk.rglob("*.wav")]
            selected_wavs = random.sample(wav_paths, self.segments_num)
            for wav_path in selected_wavs:
                batch.append(self._get_random_segment(wav_path))
                labels.append(self.speaker_ids[spk.name])

        batch = torch.stack(batch)



        return batch, labels