#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
7_Entrainement_Hybride.py

Fine-tuning d'un modèle pré-entraîné (EDSR) vers une version GAN + SAM.
Objectif : Améliorer la texture (Perceptual) et la cohérence spectrale (SAM)
tout en conservant la précision pixel (L1).
"""

import os
import argparse
import logging
import random
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Matplotlib Headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. ARCHITECTURE (Générateur & Discriminateur) ---

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.res_scale = res_scale
    def forward(self, x): return x + self.body(x) * self.res_scale

class Generator(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, n_feats=128, n_blocks=16, scale=3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_feats, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, out_ch, 3, 1, 1)
        )
    def forward(self, x):
        x = self.head(x); res = self.body(x); x = x + res
        return self.tail(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        def d_block(in_f, out_f, norm=True):
            layers = [spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1))]
            if norm: layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        self.model = nn.Sequential(
            *d_block(in_channels, 64, False),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))
        )
    def forward(self, img): return self.model(img)


# --- 2. LOSSES (VGG + SAM) ---

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        for p in self.loss_network.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))

    def forward(self, sr, hr):
        sr = torch.clamp(sr, 0, 1); hr = torch.clamp(hr, 0, 1)
        # RGB Loss
        loss_rgb = torch.nn.functional.l1_loss(
            self.loss_network((sr[:,:3]-self.mean)/self.std), 
            self.loss_network((hr[:,:3]-self.mean)/self.std)
        )
        # NIR Loss (Fake RGB)
        sr_nir = sr[:, 3:4].repeat(1, 3, 1, 1); hr_nir = hr[:, 3:4].repeat(1, 3, 1, 1)
        loss_nir = torch.nn.functional.l1_loss(
            self.loss_network((sr_nir-self.mean)/self.std),
            self.loss_network((hr_nir-self.mean)/self.std)
        )
        return loss_rgb + loss_nir

class SAMLoss(nn.Module):
    """Spectral Angle Mapper Loss : Préserve la couleur spectrale."""
    def __init__(self): 
        super().__init__()
        self.eps = 1e-6

    def forward(self, output, target):
        # Produit scalaire
        dot = torch.sum(output * target, dim=1)
        # Normes
        norm_o = torch.norm(output, dim=1)
        norm_t = torch.norm(target, dim=1)
        
        # Eviter division par zéro
        denom = norm_o * norm_t + self.eps
        cos = torch.clamp(dot / denom, -1 + self.eps, 1 - self.eps)
        
        # Angle moyen (en radians)
        return torch.mean(torch.acos(cos))


# --- 3. DATASET ---

class HybridDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.lr_dir = Path(data_dir)/"LR"; self.hr_dir = Path(data_dir)/"HR"
        self.files = sorted([f.name for f in self.lr_dir.glob("*.npy")])
        self.augment = augment

    def __len__(self): return len(self.files)

    def load_item(self, idx):
        fname = self.files[idx]
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)
        # Norm / 10000
        if lr.max() > 10.0: lr /= 10000.0
        if hr.max() > 10.0: hr /= 10000.0
        return np.nan_to_num(lr), np.nan_to_num(hr), fname

    def __getitem__(self, idx):
        lr, hr, _ = self.load_item(idx)
        if self.augment:
            if random.random()<0.5: lr=np.flip(lr,2).copy(); hr=np.flip(hr,2).copy()
            if random.random()<0.5: lr=np.flip(lr,1).copy(); hr=np.flip(hr,1).copy()
            if random.random()<0.5: k=random.randint(1,3); lr=np.rot90(lr,k,(1,2)).copy(); hr=np.rot90(hr,k,(1,2)).copy()
        return torch.from_numpy(lr), torch.from_numpy(hr)
    
    def get_visu(self, idx):
        lr, hr, n = self.load_item(idx)
        return torch.from_numpy(lr), torch.from_numpy(hr), n

def save_viz(model, samples, epoch, path, device):
    model.eval()
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3*len(samples)))
    with torch.no_grad():
        for i, (lr, hr, n) in enumerate(samples):
            sr = model(lr.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0,1)
            
            def rgb(t): 
                im = t.numpy().transpose(1,2,0)[...,[2,1,0]]
                p2, p98 = np.percentile(im, (2,98))
                return np.clip((im-p2)/(p98-p2+1e-6),0,1)**(1/1.3)

            axes[i,0].imshow(rgb(lr)); axes[i,1].imshow(rgb(sr)); axes[i,2].imshow(rgb(hr))
            axes[i,0].set_title("Input"); axes[i,1].set_title("Hybrid SR"); axes[i,2].set_title("Target")
            for ax in axes[i]: ax.axis('off')
    plt.tight_layout(); plt.savefig(path); plt.close()


# --- 4. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_final_npy")
    parser.add_argument("--out_dir", default="resultats_training_hybrid")
    parser.add_argument("--pretrained", required=True, help="Chemin vers le modèle EDSR (best_model.pth)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5) # LR très bas pour le fine-tuning
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_percep", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.005)
    parser.add_argument("--lambda_sam", type=float, default=0.1) # SAM weighting
    args = parser.parse_args()

    device = torch.device("cuda")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir)/"visu").mkdir(exist_ok=True)

    # Dataset
    ds = HybridDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    vis_samples = [ds.get_visu(i) for i in random.sample(range(len(ds)), 4)]

    # Models
    gen = Generator(n_feats=128, n_blocks=16).to(device) # On suppose EDSR 128/16
    disc = Discriminator().to(device)
    
    # CHARGEMENT EDSR (SMART LOAD)
    logger.info(f"Chargement du pré-entraîné : {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    # Filtrage des clés pour éviter les erreurs
    gen_dict = gen.state_dict()
    pretrained_dict = {k: v for k, v in state.items() if k in gen_dict and v.shape == gen_dict[k].shape}
    gen.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Poids chargés : {len(pretrained_dict)} / {len(gen_dict)} layers.")

    # Optimizers
    opt_G = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Losses
    crit_l1 = nn.L1Loss()
    crit_gan = nn.BCEWithLogitsLoss()
    crit_vgg = VGGPerceptualLoss(device)
    crit_sam = SAMLoss()
    
    scaler = torch.amp.GradScaler('cuda')

    logger.info("Démarrage Fine-tuning Hybride...")

    for ep in range(1, args.epochs + 1):
        gen.train(); disc.train()
        sam_acc = 0
        
        pbar = tqdm(dl, desc=f"Ep {ep}")
        for lr, hr in pbar:
            lr = lr.to(device); hr = hr.to(device)
            
            # --- Train D ---
            opt_D.zero_grad()
            with torch.amp.autocast('cuda'):
                fake = gen(lr)
                pred_real = disc(hr)
                pred_fake = disc(fake.detach())
                
                # Relativistic GAN Loss
                loss_real = crit_gan(pred_real - pred_fake.mean(), torch.ones_like(pred_real))
                loss_fake = crit_gan(pred_fake - pred_real.mean(), torch.zeros_like(pred_fake))
                loss_d = (loss_real + loss_fake) / 2
            
            scaler.scale(loss_d).backward()
            scaler.step(opt_D)
            scaler.update()

            # --- Train G ---
            opt_G.zero_grad()
            with torch.amp.autocast('cuda'):
                fake = gen(lr)
                
                # Pixel & SAM
                l_pix = crit_l1(fake, hr)
                l_sam = crit_sam(fake, hr)
                
                # Adv & Percep
                pred_real = disc(hr).detach()
                pred_fake = disc(fake)
                l_adv = (crit_gan(pred_fake - pred_real.mean(), torch.ones_like(pred_fake)) + 
                         crit_gan(pred_real - pred_fake.mean(), torch.zeros_like(pred_real))) / 2
                l_percep = crit_vgg(fake, hr)
                
                loss_g = (l_pix * args.lambda_l1) + \
                         (l_sam * args.lambda_sam) + \
                         (l_percep * args.lambda_percep) + \
                         (l_adv * args.lambda_adv)
            
            scaler.scale(loss_g).backward()
            scaler.step(opt_G)
            scaler.update()
            
            sam_acc += l_sam.item()
            pbar.set_postfix(SAM=f"{l_sam.item():.3f}", D=f"{loss_d.item():.3f}")

        # Logs & Save
        avg_sam = sam_acc / len(dl)
        logger.info(f"Epoch {ep} | Avg SAM Loss: {avg_sam:.4f}")
        
        torch.save(gen.state_dict(), Path(args.out_dir)/"last_G.pth")
        if ep % 5 == 0:
            save_viz(gen, vis_samples, ep, Path(args.out_dir)/"visu"/f"ep_{ep:03d}.png", device)

if __name__ == "__main__":
    main()