#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4_Applique_EDSR.py

Application du modèle de Super-Resolution (Inference) sur des images satellitaires géoréférencées.
Utilise une méthode de "Sliding Window" avec recouvrement (Overlap) et pondération (Hanning)
pour éviter les artefacts de bords entre les tuiles.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Generator

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Gestion conditionnelle de Rasterio
try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
except ImportError:
    print("Erreur: La librairie 'rasterio' est manquante. Installez-la avec : pip install rasterio")
    sys.exit(1)

# Configuration Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- ARCHITECTURE MODÈLE (Doit être identique à l'entraînement) ---

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSRLite(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, n_feats=64, n_blocks=8, scale=3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_feats, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, out_ch, 3, 1, 1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x


# --- OUTILS DE TILING (DÉCOUPAGE) ---

def generate_grid(H: int, W: int, tile_size: int, overlap: int) -> Generator[Tuple[int, int, int, int], None, None]:
    """Génère les coordonnées des tuiles avec recouvrement."""
    stride = tile_size - overlap
    ys = list(range(0, max(1, H - tile_size + 1), stride))
    xs = list(range(0, max(1, W - tile_size + 1), stride))
    
    # Assurer de couvrir les bords droits et bas
    if ys[-1] != H - tile_size: ys.append(max(0, H - tile_size))
    if xs[-1] != W - tile_size: xs.append(max(0, W - tile_size))
    
    for y in ys:
        for x in xs:
            yield y, y + tile_size, x, x + tile_size

def get_blending_mask(tile_size: int, overlap: int) -> np.ndarray:
    """Crée un masque de pondération 2D (fenêtre de Hanning) pour lisser les joints."""
    if overlap <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Création d'une rampe douce sur les bords (Hanning window)
    # On crée une fenêtre 1D qu'on étend en 2D
    # Note: ceci est une implémentation simplifiée efficace
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # On réduit le poids sur les bords de la taille de l'overlap
    ramp = np.linspace(0, 1, overlap)
    
    # Bords Gauche/Haut
    mask[:overlap, :] *= ramp[:, None] # Haut
    mask[:, :overlap] *= ramp[None, :] # Gauche
    
    # Bords Droite/Bas
    mask[-overlap:, :] *= ramp[::-1, None] # Bas
    mask[:, -overlap:] *= ramp[None, ::-1] # Droite
    
    return mask


# --- CŒUR DU TRAITEMENT ---

def process_image(src_path: Path, dst_path: Path, model: nn.Module, device: torch.device, args):
    """Lit, super-résout et sauvegarde une image."""
    
    with rasterio.open(src_path) as src:
        # Lecture des bandes
        bands_idx = [int(b) for b in args.bands.split(",")]
        # Note: rasterio utilise 1-based indexing
        data = src.read([b + 1 for b in bands_idx])
        
        meta = src.meta.copy()
        transform = src.transform
    
    # Nettoyage NaN (Important pour éviter la propagation d'erreurs)
    data = np.nan_to_num(data, nan=0.0)
    
    C, H, W = data.shape
    scale = args.scale
    
    # Dimensions de sortie
    H_out, W_out = H * scale, W * scale
    
    # Buffers d'accumulation (Image finale + Poids)
    output_sum = np.zeros((C, H_out, W_out), dtype=np.float32)
    weight_sum = np.zeros((1, H_out, W_out), dtype=np.float32)
    
    # Préparation masque de mélange
    # Le masque s'applique sur la sortie HR, donc on multiplie les tailles par scale
    tile_hr = args.tile_size * scale
    overlap_hr = args.overlap * scale
    blend_mask = get_blending_mask(tile_hr, overlap_hr)
    
    # Boucle sur les tuiles
    grid = list(generate_grid(H, W, args.tile_size, args.overlap))
    
    with torch.no_grad():
        for y0, y1, x0, x1 in tqdm(grid, desc=f"Processing {src_path.name}", leave=False):
            # 1. Extraction LR
            patch_lr = data[:, y0:y1, x0:x1]
            
            # Padding si on est au bord et que le patch est plus petit que prévu
            # (Normalement géré par generate_grid, mais sécurité)
            if patch_lr.shape[1] != args.tile_size or patch_lr.shape[2] != args.tile_size:
                pad_h = args.tile_size - patch_lr.shape[1]
                pad_w = args.tile_size - patch_lr.shape[2]
                patch_lr = np.pad(patch_lr, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
            
            # 2. Préparation Tensor
            t_lr = torch.from_numpy(patch_lr).unsqueeze(0).to(device).float()
            
            # Normalisation (si le modèle a appris sur 0-1)
            t_lr /= args.norm_val
            
            # 3. Inférence (AMP)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                t_sr = model(t_lr)
            
            # Dénormalisation + Clamp
            t_sr = t_sr.clamp(0, 1.5) * args.norm_val # Clamp 1.5 pour tolérer légère radiance excessive
            patch_sr = t_sr.squeeze(0).cpu().numpy()
            
            # 4. Placement dans l'image finale
            # Coordonnées HR
            y0_hr, x0_hr = y0 * scale, x0 * scale
            
            # On découpe le patch SR généré pour ne garder que la partie valide (si padding)
            valid_h = (y1 - y0) * scale
            valid_w = (x1 - x0) * scale
            patch_sr = patch_sr[:, :valid_h, :valid_w]
            
            # Masque local ajusté à la taille valide
            local_mask = blend_mask[:valid_h, :valid_w]
            
            output_sum[:, y0_hr:y0_hr+valid_h, x0_hr:x0_hr+valid_w] += patch_sr * local_mask
            weight_sum[:, y0_hr:y0_hr+valid_h, x0_hr:x0_hr+valid_w] += local_mask

    # Normalisation finale par les poids
    final_image = output_sum / (weight_sum + 1e-8)
    
    # Mise à jour Géoréférencement (Résolution divisée par scale)
    new_transform = Affine(transform.a / scale, transform.b, transform.c,
                           transform.d, transform.e / scale, transform.f)
    
    meta.update({
        "height": H_out,
        "width": W_out,
        "transform": new_transform,
        "dtype": "float32",
        "compress": "deflate" # Compression pour économiser disque
    })
    
    # Sauvegarde
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(final_image)


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Inférence Super-Resolution")
    parser.add_argument("--src_dir", required=True, help="Dossier contenant les images L8")
    parser.add_argument("--dst_dir", required=True, help="Dossier de sortie")
    parser.add_argument("--checkpoint", required=True, help="Chemin vers le modèle .pth")
    
    # Paramètres Modèle
    parser.add_argument("--n_feats", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=16)
    parser.add_argument("--scale", type=int, default=3)
    
    # Paramètres Traitement
    parser.add_argument("--tile_size", type=int, default=128, help="Taille tuile entrée (LR)")
    parser.add_argument("--overlap", type=int, default=16, help="Taille recouvrement entrée (LR)")
    parser.add_argument("--bands", type=str, default="1,2,3,4", help="Indices bandes (0-based) à traiter")
    parser.add_argument("--norm_val", type=float, default=10000.0, help="Valeur max pour normalisation")
    
    args = parser.parse_args()

    # Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Chargement modèle depuis {args.checkpoint}...")
    
    # Chargement Modèle
    model = EDSRLite(in_ch=4, out_ch=4, n_feats=args.n_feats, n_blocks=args.n_blocks, scale=args.scale)
    try:
        state = torch.load(args.checkpoint, map_location=device)
        # Gestion si 'state_dict' est imbriqué ou non
        if 'model' in state: state = state['model']
        model.load_state_dict(state, strict=True)
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        return

    model.to(device)
    model.eval()

    # Recherche Fichiers
    files = list(src_root.glob("*.tif"))
    if not files:
        logger.warning(f"Aucun fichier .tif trouvé dans {src_root}")
        return

    logger.info(f"Traitement de {len(files)} images sur {device}...")

    # Boucle principale
    for f in tqdm(files, desc="Global Progress"):
        # Nom de sortie : Image_L8.tif -> Image_SR.tif
        out_name = f.name.replace("L8", "SR").replace("LR", "SR")
        if out_name == f.name: out_name = f"SR_{f.name}" # Fallback
        
        dst_path = dst_root / out_name
        
        try:
            process_image(f, dst_path, model, device, args)
        except Exception as e:
            logger.error(f"Echec sur {f.name}: {e}")

    logger.info("Terminé.")

if __name__ == "__main__":
    main()