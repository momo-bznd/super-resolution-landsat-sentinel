#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3_Entrainement_EDSR.py

Entraînement d'un modèle Super-Resolution (EDSR-Lite) sur données patchées (.npy).
Supporte :
- Mixed Precision (AMP) pour accélération A100/V100/RTX.
- Visualisation automatique (Grid LR/SR/HR) à chaque époque.
- Sauvegarde du meilleur modèle (Best Loss).
"""

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional

# Configuration Matplotlib pour serveur headless (sans écran)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- ARCHITECTURE MODÈLE (EDSR Lite) ---

class ResBlock(nn.Module):
    """Bloc résiduel standard pour EDSR."""
    def __init__(self, n_feats: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale

class EDSRLite(nn.Module):
    """
    Modèle EDSR allégé pour Super-Resolution x3 (par défaut).
    """
    def __init__(self, n_feats: int = 64, n_blocks: int = 16, scale: int = 3):
        super().__init__()
        self.head = nn.Conv2d(4, n_feats, 3, 1, 1) # 4 canaux (R,G,B,NIR) ou autres
        
        self.body = nn.Sequential(*[
            ResBlock(n_feats) for _ in range(n_blocks)
        ])
        
        # Upsampling module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 4, 3, 1, 1) # Sortie 4 canaux
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x


# --- GESTION DES DONNÉES ---

class NpyDataset(Dataset):
    """Dataset optimisé pour lire les fichiers .npy générés par l'étape 2."""
    def __init__(self, root_dir: Path, augment: bool = True):
        self.lr_dir = root_dir / "LR"
        self.hr_dir = root_dir / "HR"
        self.files = sorted([f.name for f in self.lr_dir.glob("*.npy")])
        self.augment = augment
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"Aucun fichier .npy trouvé dans {self.lr_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        
        # Chargement (supposé float32 ou uint16)
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)
        
        # Normalisation S2 (0-10000 -> 0-1) si nécessaire
        if lr.max() > 100.0: lr /= 10000.0
        if hr.max() > 100.0: hr /= 10000.0
        
        # Sécurité NaN
        lr = np.nan_to_num(lr, nan=0.0)
        hr = np.nan_to_num(hr, nan=0.0)

        # Augmentation Data (Flip/Rotate)
        if self.augment:
            if random.random() < 0.5:
                lr = np.flip(lr, axis=2).copy(); hr = np.flip(hr, axis=2).copy()
            if random.random() < 0.5:
                lr = np.flip(lr, axis=1).copy(); hr = np.flip(hr, axis=1).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                lr = np.rot90(lr, k, axes=(1, 2)).copy(); hr = np.rot90(hr, k, axes=(1, 2)).copy()

        return torch.from_numpy(lr), torch.from_numpy(hr)

    def get_visu_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Récupère un sample brut sans augmentation pour la visualisation."""
        fname = self.files[idx]
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)
        
        if lr.max() > 100.0: lr /= 10000.0
        if hr.max() > 100.0: hr /= 10000.0
        
        return torch.from_numpy(lr), torch.from_numpy(hr), fname


# --- VISUALISATION ---

def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """Convertit un tenseur (C,H,W) en image RGB (H,W,C) affichable."""
    img = t.cpu().numpy().transpose(1, 2, 0) # CHW -> HWC
    
    # Extraction RGB (Bandes 2,1,0 supposées R,G,B)
    rgb = img[..., [2, 1, 0]]
    
    # Robust Auto-Level (Percentile clipping)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    rgb = np.clip(rgb, 0, 1)
    
    # Gamma correction légère pour visibilité
    return np.power(rgb, 1/1.2)

def save_validation_grid(model, samples, epoch, out_path, device):
    """Génère et sauvegarde une grille comparative (Input | Prediction | Target)."""
    model.eval()
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3 * rows))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    fig.suptitle(f"Epoch {epoch} - Super Resolution", fontsize=14, y=0.95)

    with torch.no_grad():
        for i, (lr, hr, name) in enumerate(samples):
            # Inference
            lr_dev = lr.unsqueeze(0).to(device)
            sr = model(lr_dev).squeeze(0).cpu()

            # Images RGB
            img_lr = tensor_to_rgb(lr)
            img_sr = tensor_to_rgb(sr)
            img_hr = tensor_to_rgb(hr)

            # Affichage
            lbls = ["LR Input", f"SR Output (Ep {epoch})", "HR Target"]
            imgs = [img_lr, img_sr, img_hr]
            
            for j in range(3):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.imshow(imgs[j])
                ax.axis('off')
                if i == 0: ax.set_title(lbls[j], fontsize=10, fontweight='bold')
                if j == 0: ax.text(-0.1, 0.5, name, transform=ax.transAxes, 
                                   va='center', rotation=90, fontsize=8)

    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# --- MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser(description="Entrainement EDSR Pro")
    parser.add_argument("--data_dir", type=str, default="data_prepared", help="Dossier contenant LR/ et HR/")
    parser.add_argument("--out_dir", type=str, default="checkpoints_edsr", help="Dossier de sortie")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_feats", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=16)
    parser.add_argument("--scale", type=int, default=3, help="Facteur d'échelle (doit matcher le dataset)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--visu", action='store_true', default=True, help="Activer la visualisation")
    args = parser.parse_args()

    # Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "visu").mkdir(exist_ok=True)

    logger.info(f"Démarrage sur {device}. Batch={args.batch_size}, Feats={args.n_feats}")

    # Dataset & Split
    full_dataset = NpyDataset(Path(args.data_dir), augment=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                        num_workers=args.workers, pin_memory=True)

    # Samples fixes pour visualisation (depuis le set de validation)
    fixed_samples = []
    if args.visu:
        indices = list(range(len(val_ds)))
        random.shuffle(indices)
        # On prend 3 samples au hasard
        for idx in indices[:3]:
            # Accès via dataset sous-jacent car val_ds est un Subset
            real_idx = val_ds.indices[idx]
            fixed_samples.append(full_dataset.get_visu_sample(real_idx))

    # Model, Opt, Loss
    model = EDSRLite(n_feats=args.n_feats, n_blocks=args.n_blocks, scale=args.scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                              steps_per_epoch=len(train_dl), epochs=args.epochs)
    criterion = nn.L1Loss()
    scaler = GradScaler() # Pour Mixed Precision

    best_loss = float('inf')

    # Boucle d'époques
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        
        for lr, hr in pbar:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward Pass (AMP)
            with autocast(device_type='cuda', enabled=(device.type=='cuda')):
                sr = model(lr)
                loss = criterion(sr, hr)

            if not torch.isfinite(loss):
                logger.warning("Perte infinie/NaN détectée. Skip batch.")
                continue

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_train_loss = epoch_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_dl:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                val_loss += criterion(sr, hr).item()
        
        avg_val_loss = val_loss / len(val_dl)
        
        # Logging & Save
        msg = f"Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}"
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), out_root / "best_model.pth")
            msg += " | ⭐ Best Saved"
        
        torch.save(model.state_dict(), out_root / "last_model.pth")
        logger.info(msg)

        # Visualisation
        if args.visu and fixed_samples:
            vis_path = out_root / "visu" / f"epoch_{epoch:03d}.png"
            save_validation_grid(model, fixed_samples, epoch, vis_path, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur. Arrêt.")