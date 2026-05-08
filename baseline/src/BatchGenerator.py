import random
import torch
from torch.utils.data import Dataset
import librosa
from . import AudioAugment, download_dataset


class BatchGenerator(Dataset):
    def __init__(
        self,
        dataset_paths: download_dataset.DatasetPaths,
        sample_rate=16000,
        segment_len=3.0,
        speakers_num=32,
        segments_num=6,
        max_unique: int | None = 100,
    ):
        self.dataset_paths = dataset_paths
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.speakers_num = speakers_num
        self.segments_num = segments_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get all different speakers
        train_dataset_path = self.dataset_paths.train_dataset
        all_speakers = [spk for spk in train_dataset_path.iterdir() if spk.is_dir()]
        self.set_train_speaker_paths(
            random.sample(all_speakers, k=min(max_unique, len(all_speakers)))
            if max_unique
            else all_speakers
        )

        # data augmentation
        self.augment = AudioAugment.AudioAugment(
            self.dataset_paths.noise_dataset, self.dataset_paths.reverb_dataset
        )

        # Load Validation Protocol (if provided)
        self._load_evaluation_pairs()

    def _load_evaluation_pairs(self):
        self.evaluation_pairs = []
        eval_pairs_path = self.dataset_paths.evaluation_pairs
        base_eval_dir = self.dataset_paths.evaluation_dataset

        if not eval_pairs_path.exists():
            return

        with open(eval_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    label = int(parts[0])
                    path1 = base_eval_dir / parts[1]
                    path2 = base_eval_dir / parts[2]
                    self.evaluation_pairs.append((label, path1, path2))

    def set_train_speaker_paths(self, new_speaker_paths):
        self.speaker_paths = new_speaker_paths
        self.speaker_ids = {
            spk.name: index for index, spk in enumerate(self.speaker_paths)
        }
        self.total_unique_train_speakers = len(self.speaker_paths)

        # print(self.speaker_ids)

    def _load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
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
            wav = wav[start : start + seg_len]

        return wav

    def generate_random_speaker_balanced_batch(self):

        selected_spks = random.sample(
            self.speaker_paths, min(len(self.speaker_paths), self.speakers_num)
        )

        batch = []
        labels = []

        for spk in selected_spks:
            wav_paths = [f for f in spk.rglob("*.wav")]
            selected_wavs = random.sample(
                wav_paths, min(len(wav_paths), self.segments_num)
            )
            for wav_path in selected_wavs:
                batch.append(self._get_random_segment(wav_path))
                labels.append(self.speaker_ids[spk.name])

        batch = torch.stack(batch).to(self.device)

        augmented_batch = self.augment(batch.unsqueeze(1)).squeeze(1)

        return augmented_batch, labels

    def get_evaluation_batch(self, batch_size=32, start_idx=0):
        assert start_idx + batch_size < len(self.evaluation_pairs)

        pairs = self.evaluation_pairs[start_idx : start_idx + batch_size]
        wavs_a, wavs_b, labels = [], [], []

        for label, p1, p2 in pairs:
            wavs_a.append(self._get_random_segment(p1))
            wavs_b.append(self._get_random_segment(p2))
            labels.append(label)

        return (
            torch.stack(wavs_a),
            torch.stack(wavs_b),
            torch.tensor(labels),
        )
