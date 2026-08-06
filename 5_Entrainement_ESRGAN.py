#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5_Entrainement_ESRGAN.py

Entraînement d'un modèle ESRGAN (Generative Adversarial Network) pour la Super-Resolution.
Particularités :
- Warmup L1 : 30 époques de simple SR (comme EDSR) pour stabiliser.
- GAN Training : Ensuite, activation du Discriminateur + Perceptual Loss (VGG).
- Support 4 Canaux : RGB + NIR (Le NIR est traité spécifiquement pour la Perceptual Loss).
"""

import os
import argparse
import logging
import random
import contextlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Matplotlib Headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- ARCHITECTURE GÉNÉRATEUR (RRDBNet simplifié) ---

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale

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
        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)


# --- ARCHITECTURE DISCRIMINATEUR (VGG-Style) ---

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        
        def d_block(in_f, out_f, norm=True):
            layers = [spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1))] # Stride 2
            if norm: layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *d_block(in_channels, 64, False),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            spectral_norm(nn.Conv2d(512, 1, 3, 1, 1)) # PatchGAN output
        )

    def forward(self, img):
        return self.model(img)


# --- LOSSES (PERCEPTUAL VGG) ---

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Chargement VGG19 pré-entraîné (features seulement)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # On garde jusqu'à la couche 35 (avant le dernier MaxPool)
        self.loss_network = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        for p in self.loss_network.parameters(): p.requires_grad = False
        
        # Normalisation ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))

    def forward(self, sr, hr):
        # Clip 0-1 pour stabilité
        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        # 1. Perte sur RGB
        sr_rgb = (sr[:, :3] - self.mean) / self.std
        hr_rgb = (hr[:, :3] - self.mean) / self.std
        loss_rgb = F.l1_loss(self.loss_network(sr_rgb), self.loss_network(hr_rgb))

        # 2. Perte sur NIR (Dupliqué en 3 canaux pour simuler du Gris RGB)
        sr_nir = sr[:, 3:4].repeat(1, 3, 1, 1)
        hr_nir = hr[:, 3:4].repeat(1, 3, 1, 1)
        sr_nir = (sr_nir - self.mean) / self.std
        hr_nir = (hr_nir - self.mean) / self.std
        loss_nir = F.l1_loss(self.loss_network(sr_nir), self.loss_network(hr_nir))

        return loss_rgb + loss_nir


# --- DATASET ---

class EsrganDataset(Dataset):
    def __init__(self, data_dir: Path, augment: bool = True):
        self.lr_dir = data_dir / "LR"
        self.hr_dir = data_dir / "HR"
        self.files = sorted([f.name for f in self.lr_dir.glob("*.npy")])
        self.augment = augment

    def __len__(self): return len(self.files)

    def load_item(self, idx):
        fname = self.files[idx]
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)
        
        # Normalisation S2 (0-10000 -> 0-1)
        if lr.max() > 10.0: lr /= 10000.0
        if hr.max() > 10.0: hr /= 10000.0
        
        # NaN safety
        lr = np.nan_to_num(lr, nan=0.0)
        hr = np.nan_to_num(hr, nan=0.0)

        return lr, hr, fname

    def __getitem__(self, idx):
        lr, hr, _ = self.load_item(idx)
        
        if self.augment:
            if random.random() < 0.5:
                lr = np.flip(lr, 2).copy(); hr = np.flip(hr, 2).copy()
            if random.random() < 0.5:
                lr = np.flip(lr, 1).copy(); hr = np.flip(hr, 1).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                lr = np.rot90(lr, k, (1,2)).copy(); hr = np.rot90(hr, k, (1,2)).copy()

        return torch.from_numpy(lr), torch.from_numpy(hr)

    def get_visu_sample(self, idx):
        lr, hr, name = self.load_item(idx)
        return torch.from_numpy(lr), torch.from_numpy(hr), name


# --- VISU ---

def save_viz(model, samples, epoch, path, device):
    model.eval()
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
    
    def to_rgb(t):
        if isinstance(t, torch.Tensor): t = t.cpu().numpy()
        rgb = t.transpose(1, 2, 0)[..., [2,1,0]] # CHW -> HWC, RGB
        rgb = np.nan_to_num(rgb)
        p2, p98 = np.percentile(rgb, (2, 98))
        if p98 > p2: rgb = (rgb - p2) / (p98 - p2)
        return np.clip(rgb, 0, 1) ** (1/1.3)

    with torch.no_grad():
        for i, (lr, hr, name) in enumerate(samples):
            sr = model(lr.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
            
            axes[i, 0].imshow(to_rgb(lr)); axes[i,0].axis('off')
            axes[i, 1].imshow(to_rgb(sr)); axes[i,1].axis('off')
            axes[i, 2].imshow(to_rgb(hr)); axes[i,2].axis('off')
            
            if i == 0:
                axes[i, 0].set_title("Input (LR)")
                axes[i, 1].set_title(f"ESRGAN (Ep {epoch})")
                axes[i, 2].set_title("Target (HR)")
    
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_final_npy")
    parser.add_argument("--out_dir", default="resultats_training_esrgan")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30, help="Epoques de chauffe (L1 only)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_feats", type=int, default=128) # Plus gros que EDSR
    parser.add_argument("--n_blocks", type=int, default=16)
    # Poids des pertes
    parser.add_argument("--lambda_l1", type=float, default=0.01)
    parser.add_argument("--lambda_percep", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visu").mkdir(exist_ok=True)

    # Dataset
    ds = EsrganDataset(Path(args.data_dir))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    
    # Samples Visu
    vis_indices = random.sample(range(len(ds)), 4)
    vis_samples = [ds.get_visu_sample(i) for i in vis_indices]

    # Models
    gen = Generator(n_feats=args.n_feats, n_blocks=args.n_blocks).to(device)
    disc = Discriminator().to(device)
    
    # Optimizers
    opt_G = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Criterions
    crit_l1 = nn.L1Loss()
    crit_gan = nn.BCEWithLogitsLoss()
    crit_vgg = VGGPerceptualLoss(device)
    
    scaler = torch.amp.GradScaler('cuda')

    logger.info(f"Start ESRGAN. Warmup: {args.warmup} eps. Total: {args.epochs} eps.")

    for epoch in range(1, args.epochs + 1):
        gen.train(); disc.train()
        
        # Mode GAN activé ?
        use_gan = (epoch > args.warmup)
        
        # Réduction LR après warmup
        if epoch == args.warmup + 1:
            logger.info("Fin du Warmup -> Activation GAN & LR / 2")
            for pg in opt_G.param_groups: pg['lr'] *= 0.5
            for pg in opt_D.param_groups: pg['lr'] *= 0.5

        pbar = tqdm(dl, desc=f"Ep {epoch} {'(GAN)' if use_gan else '(Warmup)'}")
        
        for lr, hr in pbar:
            lr = lr.to(device); hr = hr.to(device)
            
            # --- 1. Train Discriminator (Seulement si GAN activé) ---
            loss_d = torch.tensor(0.0, device=device)
            
            if use_gan:
                opt_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    fake = gen(lr)
                    # Relativistic GAN
                    real_pred = disc(hr)
                    fake_pred = disc(fake.detach())
                    
                    real_ra = real_pred - fake_pred.mean()
                    fake_ra = fake_pred - real_pred.mean()
                    
                    l_d_real = crit_gan(real_ra, torch.ones_like(real_ra))
                    l_d_fake = crit_gan(fake_ra, torch.zeros_like(fake_ra))
                    loss_d = (l_d_real + l_d_fake) / 2
                
                scaler.scale(loss_d).backward()
                scaler.step(opt_D)
                scaler.update()

            # --- 2. Train Generator ---
            opt_G.zero_grad()
            with torch.amp.autocast('cuda'):
                fake = gen(lr)
                
                # Pixel Loss
                l_pixel = crit_l1(fake, hr)
                
                if use_gan:
                    # Perceptual
                    l_percep = crit_vgg(fake, hr)
                    
                    # Adversarial (Relativistic)
                    real_pred = disc(hr).detach()
                    fake_pred = disc(fake)
                    l_adv_fake = crit_gan(fake_pred - real_pred.mean(), torch.ones_like(fake_pred))
                    l_adv_real = crit_gan(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))
                    l_adv = (l_adv_fake + l_adv_real) / 2
                    
                    loss_g = (l_pixel * args.lambda_l1) + (l_percep * args.lambda_percep) + (l_adv * args.lambda_adv)
                else:
                    loss_g = l_pixel
            
            scaler.scale(loss_g).backward()
            scaler.step(opt_G)
            scaler.update()
            
            pbar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}")

        # Save & Visu
        torch.save(gen.state_dict(), out_dir / "last_G.pth")
        if epoch % 5 == 0 or epoch == 1:
            save_viz(gen, vis_samples, epoch, out_dir / "visu" / f"ep_{epoch:03d}.png", device)

if __name__ == "__main__":
    main()