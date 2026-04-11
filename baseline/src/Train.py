import BatchGenerator
import AudioAugment
import FeatureExtractor
import EmbeddingModel
from eval_metrics import eer_metric, minDCF_metric
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict
import torch.nn.functional as F
import numpy as np


batch_generator = BatchGenerator.BatchGenerator("../Data/subset_100_spks")
#augment = AudioAugment.AudioAugment()
feature_extractor = FeatureExtractor.FeatureExtractor()
embed_model = EmbeddingModel.EmbeddingModel()
criterion = nn.CrossEntropyLoss()

optimizer = AdamW(embed_model.parameters(), lr=2e-5, weight_decay=0.01)

ITER_NUM = 1000
EVAL_INTERVAL = 5

MODEL_DIR = "model"
MODEL_NAME = "checkpoint_"+str(ITER_NUM)

model_dir = Path(MODEL_DIR)
model_dir.mkdir(parents=True, exist_ok=True)

model_path = Path(MODEL_DIR+"/"+MODEL_NAME+".pt")
if model_path.exists():
    checkpoint = torch.load(model_path, weights_only=False)
    embed_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    old_log = checkpoint["log"]
else:
    optimizer = AdamW(embed_model.parameters(), lr=2e-5, weight_decay=0.01)

log = {
    "loss_history" : [],
    "grad_norm_history" : [],
    "loss_history_EMA" : [],
    "same_spk_similarity": [],
    "different_spk_similarity" : [],
    "margin" : []
}

def show_and_save_figs(log):
    plt.figure(figsize=(10, 5))
    plt.plot(torch.tensor(log["loss_history"]), label='Trénovací loss')
    plt.title('Loss')
    plt.xlabel('Iterace')
    plt.ylabel('Loss hodnota')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR+"/"+MODEL_NAME+"_loss.png") 
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(torch.tensor(log["loss_history_EMA"]), label='EMA')
    plt.title('EMA')
    plt.xlabel('Iterace')
    plt.ylabel('EMA loss hodnoty')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR+"/"+MODEL_NAME+"_EMA.png") 
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(torch.tensor(log["grad_norm_history"]), label='Gradient norm')
    plt.title('Gradient norm')
    plt.xlabel('Iterace')
    plt.ylabel('Loss hodnota')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR+"/"+MODEL_NAME+"_grad-norm.png") 
    plt.show()

    plt.plot(log["same_spk_similarity"], label="same")
    plt.plot(log["different_spk_similarity"], label="diff")
    plt.plot(log["margin"], label="margin")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("similarity")
    plt.savefig(MODEL_DIR+"/"+MODEL_NAME+"_margin.png") 
    plt.show()

show_and_save_figs(old_log)

def compute_grad_norm(model):
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5

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

for step in range(0, ITER_NUM):
    batch, labels = batch_generator.generate_random_speaker_balanced_batch()
    batch = torch.stack([feature_extractor.get_features(b) for b in batch])
    #batch = batch.float
    labels = torch.tensor(labels)
    embeddings, logits = embed_model.forward(batch)

    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    log["grad_norm_history"].append(compute_grad_norm(embed_model))
    optimizer.step()

    log["loss_history"].append(loss)
    print("Iterace č. "+str(step)+": "+str(loss))

    if step % EVAL_INTERVAL == 0:
        spk_emb_dict = defaultdict(list)
        for emb, spk in zip(embeddings, labels):
            spk_emb_dict[spk.item()].append(emb)
        same_sims = compute_same_sims(spk_emb_dict, pairs_per_spk=3)
        diff_sims = compute_diff_sims(
            spk_emb_dict,
            target_pairs=len(same_sims)
        )
        mean_same = np.mean(same_sims)
        mean_diff = np.mean(diff_sims)
        margin = mean_same - mean_diff
        log["same_spk_similarity"].append(mean_same)
        log["different_spk_similarity"].append(mean_diff)
        log["margin"].append(margin)

    # Metrics
    # TODO otestovat

    #if step == ITER_NUM - 1:
    #    spk_emb_dict = defaultdict(list)
    #    for emb, spk in zip(embeddings, labels):
    #        spk_emb_dict[spk.item()].append(emb)
    #    same_sims = compute_same_sims(spk_emb_dict, pairs_per_spk=3)
    #    diff_sims = compute_diff_sims(
    #        spk_emb_dict,
    #        target_pairs=len(same_sims)
    #    )
    #    scores = same_sims + diff_sims
    #    labels = [1]*len(same_sims) + [0]*len(diff_sims)
    #    eer, _ = eer_metric(scores, labels)
    #    min_dcf, _ = minDCF_metric(scores, labels)
    


log["loss_history_EMA"] = pd.Series(log["loss_history"]).ewm(alpha=0.1, adjust=False).mean().to_list()

torch.save({
    "model": embed_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step" : step,
    "log" : log
}, model_path)


    