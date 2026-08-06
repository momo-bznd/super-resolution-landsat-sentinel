#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2_prepare_data.py

Conversion des paires GeoTIFF (L8/S2) en patchs .npy optimisés pour l'entraînement.
Gère le découpage (tiling), le filtrage de qualité (NoData) et la sélection des bandes.
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- UTILITAIRES ---

def parse_bands(band_str: str) -> List[int]:
    """Convertit '1,2,3' en [0, 1, 2] (0-based indices)."""
    return [int(b.strip()) - 1 for b in band_str.split(',') if b.strip().isdigit()]

def find_pairs(l8_dir: Path, s2_dir: Path) -> List[Tuple[Path, Path]]:
    """Apparie les images L8 et S2 basées sur le nom de fichier (sans suffixe)."""
    l8_files = sorted(list(l8_dir.glob("*.tif")))
    pairs = []
    
    # Création d'un index S2 pour recherche rapide
    # On suppose que le nom de base est identique ou contient la clé
    s2_index = {f.name: f for f in s2_dir.glob("*.tif")}
    
    for l8_path in l8_files:
        # Tente de trouver le fichier S2 correspondant (meme nom)
        if l8_path.name in s2_index:
            pairs.append((l8_path, s2_index[l8_path.name]))
        else:
            # Fallback : gestion des suffixes potentiels (_L8 vs _S2)
            # Ex: image_L8.tif -> image_S2.tif
            potential_name = l8_path.name.replace("L8", "S2") # simple heuristique
            if potential_name in s2_index:
                 pairs.append((l8_path, s2_index[potential_name]))

    return pairs

def read_raster(path: Path, bands: List[int]) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """Lit les bandes spécifiques d'un raster."""
    try:
        with rasterio.open(path) as src:
            # rasterio indexes are 1-based, we pass list of 1-based indices
            read_bands = [b + 1 for b in bands]
            data = src.read(read_bands)
            return data, (src.height, src.width)
    except Exception as e:
        return None, (0, 0)


# --- WORKER ---

def process_scene(args) -> int:
    """Fonction worker pour découper une scène en patchs .npy."""
    (l8_path, s2_path, out_dir, patch_size, scale, 
     min_valid, bands_l8, bands_s2) = args

    # Lecture des images
    img_l8, (h_l8, w_l8) = read_raster(l8_path, bands_l8)
    img_s2, (h_s2, w_s2) = read_raster(s2_path, bands_s2)

    if img_l8 is None or img_s2 is None:
        return 0

    # Validation dimensions (tolérance de 1-2 pixels possible, ici stricte)
    if h_s2 != h_l8 * scale or w_s2 != w_l8 * scale:
        # Resize simple si léger décalage (optionnel, ici on skip pour rigueur)
        return 0

    base_name = l8_path.stem.replace("_L8", "").replace("L8", "")
    
    # Dossiers cibles
    dir_lr = out_dir / "LR"
    dir_hr = out_dir / "HR"
    
    count = 0
    stride = patch_size # Pas de chevauchement (stride = taille patch)
    
    # Boucle de découpage
    for y in range(0, h_l8 - patch_size + 1, stride):
        for x in range(0, w_l8 - patch_size + 1, stride):
            
            # Extraction LR
            crop_l8 = img_l8[:, y:y+patch_size, x:x+patch_size]
            
            # Extraction HR correspondante
            y_hr, x_hr = y * scale, x * scale
            ps_hr = patch_size * scale
            crop_s2 = img_s2[:, y_hr:y_hr+ps_hr, x_hr:x_hr+ps_hr]

            # Vérification NoData / Qualité
            # Critère : pas de NaN et ratio de pixels > 0 suffisant
            mask_valid = np.isfinite(crop_l8).all(axis=0) & (crop_l8 > 0).any(axis=0)
            if mask_valid.mean() < min_valid:
                continue
            
            if not np.isfinite(crop_s2).all():
                continue

            # Sauvegarde
            out_name = f"{base_name}_y{y}_x{x}.npy"
            np.save(dir_lr / out_name, crop_l8)
            np.save(dir_hr / out_name, crop_s2)
            count += 1
            
    return count


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Tiling L8/S2 vers NPY")
    
    # Chemins
    parser.add_argument("--l8_dir", type=str, required=True, help="Dossier source L8 (tif)")
    parser.add_argument("--s2_dir", type=str, required=True, help="Dossier source S2 (tif)")
    parser.add_argument("--out_dir", type=str, default="data_prepared", help="Dossier sortie NPY")
    
    # Paramètres images
    parser.add_argument("--patch", type=int, default=160, help="Taille du patch LR (L8)")
    parser.add_argument("--scale", type=int, default=3, help="Facteur d'échelle (3 pour L8->S2)")
    parser.add_argument("--min_valid", type=float, default=0.90, help="Ratio min de pixels valides")
    
    # Bandes (Format "1,2,3" => Indices 1-based)
    # L8 défaut: B2,B3,B4,B5 (Blue, Green, Red, NIR)
    # S2 défaut: B2,B3,B4,B8 (Blue, Green, Red, NIR)
    parser.add_argument("--bands_l8", type=str, default="2,3,4,5", help="Bandes L8 à extraire")
    parser.add_argument("--bands_s2", type=str, default="2,3,4,8", help="Bandes S2 à extraire")
    
    # Sys
    parser.add_argument("--workers", type=int, default=8, help="Nombre de processus parallèles")

    args = parser.parse_args()
    
    # Setup Paths
    root_l8 = Path(args.l8_dir)
    root_s2 = Path(args.s2_dir)
    root_out = Path(args.out_dir)
    
    (root_out / "LR").mkdir(parents=True, exist_ok=True)
    (root_out / "HR").mkdir(parents=True, exist_ok=True)
    
    # Parsing
    bands_l8_idx = parse_bands(args.bands_l8)
    bands_s2_idx = parse_bands(args.bands_s2)
    
    logger.info(f"Scan des paires dans {root_l8} et {root_s2}...")
    pairs = find_pairs(root_l8, root_s2)
    
    if not pairs:
        logger.error("Aucune paire correspondante trouvée. Vérifiez les noms de fichiers.")
        return

    logger.info(f"Début du traitement de {len(pairs)} scènes avec {args.workers} workers.")
    logger.info(f"Config: Patch={args.patch}px (LR), Scale={args.scale}, Valid>{args.min_valid}")

    # Préparation des tâches
    tasks = [
        (l8, s2, root_out, args.patch, args.scale, 
         args.min_valid, bands_l8_idx, bands_s2_idx) 
        for l8, s2 in pairs
    ]
    
    # Exécution parallèle
    total_patches = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # map préserve l'ordre, mais as_completed est souvent plus responsive pour la barre de progression
        futures = {executor.submit(process_scene, t): t for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(pairs), desc="Tiling"):
            try:
                count = future.result()
                total_patches += count
            except Exception as e:
                logger.error(f"Erreur worker: {e}")

    duration = time.time() - start_time
    logger.info(f"Terminé en {duration:.1f}s.")
    logger.info(f"Total patchs générés : {total_patches}")
    logger.info(f"Sortie : {root_out}")

if __name__ == "__main__":
    main()