import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch.nn.functional as F
import numpy as np
from . import BatchGenerator, AudioAugment, FeatureExtractor, EmbeddingModel
from .eval_metrics import eer_metric, minDCF_metric
from torch.optim import AdamW
from pathlib import Path
from collections import defaultdict


ITER_NUM = 1000
EVAL_INTERVAL = 5
SAVE_INTERVAL = 50
SPEAKER_LIMIT = 100 # None for no limit

MODEL_DIR = "model"
MODEL_NAME = "checkpoint_" + str(ITER_NUM)


def show_and_save_figs(log):
    plt.figure(figsize=(10, 5))
    plt.plot(log["loss_history"], label="Trénovací loss")
    plt.title("Loss")
    plt.xlabel("Iterace")
    plt.ylabel("Loss hodnota")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR + "/" + MODEL_NAME + "_loss.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["loss_history_EMA"], label="EMA")
    plt.title("EMA")
    plt.xlabel("Iterace")
    plt.ylabel("EMA loss hodnoty")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR + "/" + MODEL_NAME + "_EMA.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(log["grad_norm_history"], label="Gradient norm")
    plt.title("Gradient norm")
    plt.xlabel("Iterace")
    plt.ylabel("Loss hodnota")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR + "/" + MODEL_NAME + "_grad-norm.png")
    plt.show()

    plt.plot(log["same_spk_similarity"], label="same")
    plt.plot(log["different_spk_similarity"], label="diff")
    plt.plot(log["margin"], label="margin")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("similarity")
    plt.savefig(MODEL_DIR + "/" + MODEL_NAME + "_margin.png")
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
    def __init__(self, dev_dataset_dir, test_dataset_dir):
        self.dev_batch_generator = BatchGenerator.BatchGenerator(dev_dataset_dir, max_unique=SPEAKER_LIMIT)
        self.test_batch_generator = BatchGenerator.BatchGenerator(test_dataset_dir, max_unique=None)
        # augment = AudioAugment.AudioAugment()
        self.feature_extractor = FeatureExtractor.FeatureExtractor()
        self.embed_model = EmbeddingModel.EmbeddingModel(
            num_speakers=self.dev_batch_generator.total_unique_speakers
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.embed_model.parameters(), lr=2e-5, weight_decay=0.01
        )
        
        self.log = {
            "loss_history": [],
            "grad_norm_history": [],
            "loss_history_EMA": [],
            "same_spk_similarity": [],
            "different_spk_similarity": [],
            "margin": [],
        }

        # load model if exists
        model_dir = (
            Path(__file__).resolve().parent.parent.joinpath(MODEL_DIR) # baseline/MODEL_DIR/
        )  
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_dir.joinpath(f"{MODEL_NAME}.pt")
        self.last_step = -1
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.embed_model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.log = checkpoint["log"]
            self.last_step = checkpoint["step"]
            self.dev_batch_generator.set_speaker_paths(checkpoint["speaker_paths"])

            #show_and_save_figs(old_log)

    def get_batch(self, batch_generator):
        batch, labels = batch_generator.generate_random_speaker_balanced_batch()
        batch = torch.stack([self.feature_extractor.get_features(b) for b in batch])
        # batch = batch.float
        labels = torch.tensor(labels)

        return batch, labels

    def evaluate_pairs(self, embeddings, labels):
        spk_emb_dict = defaultdict(list)
        for emb, spk in zip(embeddings, labels):
            spk_emb_dict[spk.item()].append(emb)
        same_sims = compute_same_sims(spk_emb_dict, pairs_per_spk=3)
        diff_sims = compute_diff_sims(spk_emb_dict, target_pairs=len(same_sims))

        return same_sims, diff_sims
    
    def save_checkpoint(self, step, log):
        torch.save(
            {
                "model": self.embed_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "log": log,
                "speaker_paths": self.dev_batch_generator.speaker_paths
            },
            self.model_path,
        )

    def train(self):
        embed_model = self.embed_model
        optimizer = self.optimizer
        criterion = self.criterion
        log = self.log

        for step in range(self.last_step + 1, ITER_NUM):
            batch, labels = self.get_batch(self.dev_batch_generator)
            embeddings, logits = embed_model.forward(batch)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            log["grad_norm_history"].append(compute_grad_norm(embed_model))
            optimizer.step()

            log["loss_history"].append(loss.item())
            print(f"Iterace č. {step + 1}/{ITER_NUM}, loss: {loss.item()}")

            if step % EVAL_INTERVAL == 0:
                same_sims, diff_sims = self.evaluate_pairs(embeddings, labels)
                mean_same = np.mean(same_sims)
                mean_diff = np.mean(diff_sims)
                margin = mean_same - mean_diff
                log["same_spk_similarity"].append(mean_same)
                log["different_spk_similarity"].append(mean_diff)
                log["margin"].append(margin)
                
            if step % SAVE_INTERVAL == 0 or step == ITER_NUM - 1:
                self.save_checkpoint(step, log)
                       

        log["loss_history_EMA"] = (
            pd.Series(log["loss_history"]).ewm(alpha=0.1, adjust=False).mean().to_list()
        )

        show_and_save_figs(log)

        return embed_model

    # Metrics
    def evaluate(self):
        # TODO otestovat
        self.test_batch_generator.speakers_num = (
            self.test_batch_generator.total_unique_speakers
        )  # use all speakers in test part

        self.embed_model.eval()  # Set to evaluation mode and disable gradient calculation
        with torch.no_grad():
            batch, labels = self.get_batch(self.test_batch_generator)

            embeddings, logits = self.embed_model.forward(batch)
            same_sims, diff_sims = self.evaluate_pairs(embeddings, labels)

        scores = same_sims + diff_sims
        labels = [1] * len(same_sims) + [0] * len(diff_sims)
        eer, eer_threshold = eer_metric(scores, labels)
        min_dcf, dcf_threshold = minDCF_metric(scores, labels)

        print("Evaluation")
        print(f"Test EER: {eer*100:.2f}% (at threshold: {eer_threshold:.4f})")
        print(f"Test Min DCF: {min_dcf:.4f} (at threshold: {dcf_threshold:.4f})")

        self.embed_model.train()
