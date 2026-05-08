import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch_audiomentations"
)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch.nn.functional as F
import numpy as np
from . import (
    BatchGenerator,
    FeatureExtractor,
    EmbeddingModel,
    model,
    loss_function,
    download_dataset,
)
from .eval_metrics import eer_metric, minDCF_metric
from torch.optim import AdamW
from pathlib import Path
from collections import defaultdict

MAX_TRAIN_EVAL_PAIRS = 500


def show_and_save_figs(log, model_dir, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(log["loss_history"], label="Trénovací loss")
    plt.title("Loss")
    plt.xlabel("Iterace")
    plt.ylabel("Loss hodnota")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir + "/" + model_name + "_loss.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["loss_history_EMA"], label="EMA")
    plt.title("EMA")
    plt.xlabel("Iterace")
    plt.ylabel("EMA loss hodnoty")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir + "/" + model_name + "_EMA.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["grad_norm_history"], label="Gradient norm")
    plt.title("Gradient norm")
    plt.xlabel("Iterace")
    plt.ylabel("Loss hodnota")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir + "/" + model_name + "_grad-norm.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["same_spk_similarity"], label="same")
    plt.plot(log["different_spk_similarity"], label="diff")
    plt.plot(log["margin"], label="margin")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("similarity")
    plt.savefig(model_dir + "/" + model_name + "_margin.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Min DCF")
    plt.plot(log["min_dcf"], label="Min DF")
    # plt.legend()
    plt.xlabel("step")
    plt.ylabel("min_dcf")
    plt.savefig(model_dir + "/" + model_name + "_min_dfc.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("EER")
    plt.plot(log["eer"], label="EER")
    # plt.legend()
    plt.xlabel("step")
    plt.ylabel("EER")
    plt.savefig(model_dir + "/" + model_name + "_eer.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["dcf_threshold"], label="DCF Threshold")
    plt.plot(log["eer_threshold"], label="EER Threshold")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("value")
    plt.savefig(model_dir + "/" + model_name + "_thresholds.png")
    plt.show()


def compute_grad_norm(model):
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm**0.5


def compute_same_sims(spk_emb_dict, pairs_per_spk=3):
    same_sims = []

    for spk, embs in spk_emb_dict.items():
        if len(embs) < 2:
            continue

        for _ in range(pairs_per_spk):
            emb_a, emb_b = random.sample(embs, 2)
            sim = F.cosine_similarity(emb_a, emb_b, dim=0)
            same_sims.append(sim.item())

    return same_sims


def compute_diff_sims(spk_emb_dict, target_pairs):
    diff_sims = []
    speakers = list(spk_emb_dict.keys())

    while len(diff_sims) < target_pairs:
        spk_a, spk_b = random.sample(speakers, 2)

        emb_a = random.choice(spk_emb_dict[spk_a])
        emb_b = random.choice(spk_emb_dict[spk_b])

        sim = F.cosine_similarity(emb_a, emb_b, dim=0)
        diff_sims.append(sim.item())

    return diff_sims


class EmbeddingModelTrainer:
    def __init__(
        self,
        dataset_paths: download_dataset.DatasetPaths,
        speaker_limit=None,
        iter_num=2000,
        eval_interval=50,
        save_interval=10,
        embed_dim=192,
        model_dir="model",
        base_model=True,
    ):
        self.batch_generator = BatchGenerator.BatchGenerator(
            dataset_paths,
            max_unique=speaker_limit,
        )

        self.feature_extractor = FeatureExtractor.FeatureExtractor()
        # print(self.dev_batch_generator.total_unique_speakers)
        self.speakers = self.batch_generator.total_unique_train_speakers

        if base_model:
            self.device = "cpu"
            self.embed_model = EmbeddingModel.EmbeddingModel(num_speakers=self.speakers)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = AdamW(
                self.embed_model.parameters(), lr=1e-3, weight_decay=1e-4
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embed_model = model.ECAPA_TDNN(embd_dim=embed_dim).to(self.device)
            self.criterion = loss_function.AAM_loss(
                embed_dim=embed_dim, n_speakers=self.speakers, device=self.device
            ).to(self.device)
            self.optimizer = AdamW(
                list(self.embed_model.parameters()) + list(self.criterion.parameters()),
                lr=1e-3,
                weight_decay=1e-4,
            )

        self.log = {
            "loss_history": [],
            "grad_norm_history": [],
            "loss_history_EMA": [],
            "same_spk_similarity": [],
            "different_spk_similarity": [],
            "margin": [],
            "eer": [],
            "eer_threshold": [],
            "min_dcf": [],
            "dcf_threshold": [],
        }

        self.iter_num = iter_num
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.model_dir = model_dir
        self.model_name = "checkpoint_" + str(self.iter_num)

        # load model if exists
        model_dir = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath(self.model_dir)  # baseline/MODEL_DIR/
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_dir.joinpath(f"{self.model_name}.pt")
        self.last_step = -1
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.embed_model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.log = checkpoint["log"]
            self.last_step = checkpoint["step"]
            self.batch_generator.set_train_speaker_paths(checkpoint["speaker_paths"])

            # show_and_save_figs(old_log)

    def get_features_batch(self, batch, labels):
        batch = torch.stack([self.feature_extractor.get_features(b) for b in batch])
        # batch = batch.float
        labels = torch.tensor(labels)

        return batch, labels

    def evaluate_pairs(self, embeddings, labels, pairs_per_spk=3):
        spk_emb_dict = defaultdict(list)
        for emb, spk in zip(embeddings, labels):
            spk_emb_dict[spk.item()].append(emb)
        same_sims = compute_same_sims(spk_emb_dict, pairs_per_spk=pairs_per_spk)
        diff_sims = compute_diff_sims(spk_emb_dict, target_pairs=len(same_sims))

        return same_sims, diff_sims

    def save_checkpoint(self, step, log):
        torch.save(
            {
                "model": self.embed_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "log": log,
                "speaker_paths": self.batch_generator.speaker_paths,
            },
            self.model_path,
        )

    def train(self):
        embed_model = self.embed_model
        optimizer = self.optimizer
        criterion = self.criterion
        log = self.log

        for step in range(self.last_step + 1, self.iter_num):
            batch, labels = self.batch_generator.generate_random_speaker_balanced_batch()
            batch, labels = self.get_features_batch(batch, labels)
            embeddings, logits = embed_model.forward(batch)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            log["grad_norm_history"].append(compute_grad_norm(embed_model))
            optimizer.step()

            log["loss_history"].append(loss.item())
            print(f"Iterace č. {step + 1}/{self.iter_num}, loss: {loss.item()}")

            if step % self.eval_interval == 0:
                same_sims, diff_sims = self.evaluate_pairs(embeddings, labels)
                mean_same = np.mean(same_sims)
                mean_diff = np.mean(diff_sims)
                margin = mean_same - mean_diff
                log["same_spk_similarity"].append(mean_same)
                log["different_spk_similarity"].append(mean_diff)
                log["margin"].append(margin)
                eer, eer_threshold, min_dcf, dcf_threshold = self.evaluate(MAX_TRAIN_EVAL_PAIRS)
                log["eer"].append(eer)
                log["eer_threshold"].append(eer_threshold)
                log["min_dcf"].append(min_dcf)
                log["dcf_threshold"].append(dcf_threshold)

            if step % self.save_interval == 0 or step == self.iter_num - 1:
                self.save_checkpoint(step, log)

        log["loss_history_EMA"] = (
            pd.Series(log["loss_history"]).ewm(alpha=0.1, adjust=False).mean().to_list()
        )

        show_and_save_figs(log, self.model_dir, self.model_name)

        return embed_model

    def train_ECAPA(self):
        embed_model = self.embed_model
        optimizer = self.optimizer
        criterion = self.criterion
        log = self.log

        for step in range(self.last_step + 1, self.iter_num):
            batch, labels = self.batch_generator.generate_random_speaker_balanced_batch()
            batch, labels = self.get_features_batch(batch, labels)
            
            batch = batch.to(self.device)
            labels = labels.to(self.device)
            embeddings = embed_model.forward(batch)

            loss = criterion(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            log["grad_norm_history"].append(compute_grad_norm(embed_model))
            optimizer.step()

            log["loss_history"].append(loss.item())
            print(f"Iterace č. {step + 1}/{self.iter_num}, loss: {loss.item()}")

            if step % self.eval_interval == 0:
                same_sims, diff_sims = self.evaluate_pairs(embeddings, labels)
                mean_same = np.mean(same_sims)
                mean_diff = np.mean(diff_sims)
                margin = mean_same - mean_diff
                log["same_spk_similarity"].append(mean_same)
                log["different_spk_similarity"].append(mean_diff)
                log["margin"].append(margin)
                eer, eer_threshold, min_dcf, dcf_threshold = self.evaluate(MAX_TRAIN_EVAL_PAIRS)
                log["eer"].append(eer)
                log["eer_threshold"].append(eer_threshold)
                log["min_dcf"].append(min_dcf)
                log["dcf_threshold"].append(dcf_threshold)

            if step % self.save_interval == 0 or step == self.iter_num - 1:
                self.save_checkpoint(step, log)

        log["loss_history_EMA"] = (
            pd.Series(log["loss_history"]).ewm(alpha=0.1, adjust=False).mean().to_list()
        )

        show_and_save_figs(log, self.model_dir, self.model_name)

        return embed_model

    # Metrics
    def evaluate(self, max_pairs: None | int = None):
        self.embed_model.eval()  # Set to evaluation mode and disable gradient calculation
        scores = []
        labels = []
        eval_batch_size = 64
        
        max_pairs = max_pairs if max_pairs else len(self.batch_generator.evaluation_pairs)
        
        with torch.no_grad():
            for i in range(0, max_pairs, eval_batch_size):
                #get pairs
                audio_a, audio_b, batch_labels = self.batch_generator.get_evaluation_batch(
                    batch_size=eval_batch_size, 
                    start_idx=i
                )
                
                audio_a = audio_a.to(self.device)
                audio_a = torch.stack([self.feature_extractor.get_features(b) for b in audio_a])
                
                audio_b = audio_b.to(self.device)
                audio_b = torch.stack([self.feature_extractor.get_features(b) for b in audio_b])
                
                batch_labels = batch_labels.to(self.device)

                # Extract Embeddings
                emb_a = self.embed_model.forward(audio_a)
                emb_b = self.embed_model.forward(audio_b)
                
                emb_a = emb_a[0] if isinstance(emb_a, tuple) else emb_a
                emb_b = emb_b[0] if isinstance(emb_b, tuple) else emb_b
                
                # Cosine Similarity
                sim = F.cosine_similarity(emb_a, emb_b, dim=1)
                
                scores.extend(sim.cpu().numpy().tolist())
                labels.extend(batch_labels.cpu().numpy().tolist())

        eer, eer_threshold = eer_metric(scores, labels)
        min_dcf, dcf_threshold = minDCF_metric(scores, labels)

        print("Evaluation")
        print(f"EER: {eer*100:.2f}% (at threshold: {eer_threshold:.4f})")
        print(f"Min DCF: {min_dcf:.4f} (at threshold: {dcf_threshold:.4f})")

        self.embed_model.train()

        return eer, eer_threshold, min_dcf, dcf_threshold
