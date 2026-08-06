#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1_patch_extracteur.py

Pipeline d'extraction, alignement sub-pixel et filtrage qualité pour paires Landsat-8 / Sentinel-2.
Gère la création de patchs, le calcul de similarité (SSIM/ZNCC), les masques NoData et la division Train/Val.
"""

import os
import re
import glob
import shutil
import argparse
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from tqdm import tqdm
from scipy.ndimage import shift as nd_shift
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel

# Visualisation conditionnelle
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# --- CONFIGURATION CONSTANTES ---
PATCH_SIZE = 512
S2_SCALE = 3
S2_PATCH_SIZE = PATCH_SIZE * S2_SCALE  # 1536
NODATA_THR = 0.10
DEFAULT_THR = 0.60
ALIGN_UPSAMPLE = 10

# Regex pre-compile
_P_RE = re.compile(r'(?i)(?:^|[\/_\-\s])P(\d+)(?:[\/_\-\s]|$)')
_PAIR_RE = re.compile(r'(?i)pair[_\-\s]*?(\d+)')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- UTILITAIRES FICHIERS ---

def parse_ids(path: str) -> Tuple[Optional[int], Optional[int]]:
    norm = path.replace(os.sep, '/')
    p_m = _P_RE.search(norm)
    pair_m = _PAIR_RE.search(os.path.basename(norm))
    return (int(p_m.group(1)) if p_m else None, int(pair_m.group(1)) if pair_m else None)

def find_pairs(l8_dir: str, s2_dir: str) -> List[Tuple[str, List[str], int, int]]:
    """Indexe et apparie les fichiers L8 et S2 via Pxxx et pairxxx."""
    l8_files = glob.glob(os.path.join(l8_dir, "**", "*.tif"), recursive=True)
    s2_files = glob.glob(os.path.join(s2_dir, "**", "*.tif"), recursive=True)

    s2_map = {}
    for f in s2_files:
        ids = parse_ids(f)
        if all(ids):
            s2_map.setdefault(ids, []).append(f)

    pairs = []
    for l8 in l8_files:
        ids = parse_ids(l8)
        if all(ids) and ids in s2_map:
            pairs.append((l8, sorted(s2_map[ids]), ids[0], ids[1]))
    
    # Tri stable pour reproductibilité
    pairs.sort(key=lambda x: (x[2], x[3], os.path.basename(x[0])))
    return pairs


# --- TRAITEMENT D'IMAGES ---

def resize_chw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Redimensionnement channel-first sûr."""
    c = arr.shape[0]
    out = np.empty((c, h, w), dtype=np.float32)
    for i in range(c):
        out[i] = resize(arr[i], (h, w), order=1, preserve_range=True)
    return out

def get_nodata_mask(patch: np.ndarray, nodata_val: float = 0, zero_is_nodata: bool = False) -> np.ndarray:
    """Génère un masque booléen (True = invalide)."""
    mask = np.isnan(patch).any(axis=0)
    if nodata_val is not None:
        mask |= np.isclose(patch, nodata_val, atol=1e-6).any(axis=0)
    if zero_is_nodata:
        mask |= (patch == 0).all(axis=0)
    return mask

def mosaic_s2(paths: List[str], bbox: List[float], out_hw: Tuple[int, int]) -> Tuple[np.ndarray, object]:
    """Crée une mosaïque S2 à la volée sur la BBox demandée."""
    h, w = out_hw
    xres = (bbox[2] - bbox[0]) / w
    yres = (bbox[3] - bbox[1]) / h
    
    # Tentative de merge simple
    try:
        mosaic, trans = merge(paths, bounds=tuple(bbox), res=(xres, yres), nodata=0)
        # Crop de sécurité si dimensions non conformes (arrondis)
        if mosaic.shape[1:] != (h, w):
            mosaic = resize_chw(mosaic, h, w)
        return mosaic, trans
    except Exception as e:
        logger.warning(f"Erreur mosaïque S2: {e}")
        raise


# --- ALIGNEMENT & MÉTRIQUES ---

def estimate_shift(ref: np.ndarray, mov: np.ndarray, upsample: int = 10) -> Tuple[float, float]:
    """Estime le décalage (dy, dx) via corrélation de phase sur les gradients (Sobel)."""
    # Sobel pour robustesse aux changements radiométriques
    ref_g = sobel(np.mean(ref[:3], axis=0))
    mov_g = sobel(np.mean(mov[:3], axis=0))
    
    shift, _, _ = phase_cross_correlation(ref_g, mov_g, upsample_factor=upsample)
    return float(shift[0]), float(shift[1])

def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Applique le shift sub-pixel."""
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        out[i] = nd_shift(img[i], shift=(dy, dx), order=1)
    return out

def compute_metrics(l8: np.ndarray, s2: np.ndarray) -> Dict[str, float]:
    """Calcule ZNCC et SSIM sur image downscalée."""
    # S2 -> 512px
    s2_small = resize_chw(s2, l8.shape[1], l8.shape[2])
    
    # Normalisation basique
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
    
    l8_gray = norm(np.mean(l8[:3], axis=0))
    s2_gray = norm(np.mean(s2_small[:3], axis=0))
    
    score_ssim = ssim(l8_gray, s2_gray, data_range=1.0)
    
    # ZNCC simplifiée
    a, b = l8_gray.flatten(), s2_gray.flatten()
    a -= a.mean(); b -= b.mean()
    score_zncc = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    return {"ssim": score_ssim, "zncc": score_zncc}


# --- VISUALISATION ---

def save_debug_viz(l8: np.ndarray, s2: np.ndarray, path: str, title: str):
    if not HAS_MPL: return
    
    def to_rgb(x):
        rgb = x[:3].transpose(1, 2, 0)
        p2, p98 = np.percentile(rgb, (2, 98))
        return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    s2_view = resize_chw(s2, l8.shape[1], l8.shape[2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(to_rgb(l8)); ax1.set_title("L8 Target"); ax1.axis('off')
    ax2.imshow(to_rgb(s2_view)); ax2.set_title(f"S2 Aligned\n{title}"); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- PIPELINE PRINCIPAL ---

def process_patches(args):
    # Setup Dirs
    if args.overwrite and os.path.exists(args.export):
        shutil.rmtree(args.export)
    
    dirs = {
        "l8": os.path.join(args.export, "landsat8"),
        "s2": os.path.join(args.export, "sentinel2"),
        "vis": os.path.join(args.export, "inspect_vis")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # Discovery
    pairs = find_pairs(args.landsat, args.sentinel)
    logger.info(f"Paires trouvées: {len(pairs)}")
    
    catalogue = []

    for path_l8, paths_s2, p_idx, pair_idx in tqdm(pairs, desc="Processing"):
        with rasterio.open(path_l8) as src_l8:
            h, w = src_l8.height, src_l8.width
            meta = src_l8.meta.copy()
            trans = src_l8.transform

            for r in range(0, h, PATCH_SIZE):
                for c in range(0, w, PATCH_SIZE):
                    if r + PATCH_SIZE > h or c + PATCH_SIZE > w: continue
                    
                    base_name = f"P{p_idx}_pair{pair_idx}_r{r}_c{c}"
                    out_l8_path = os.path.join(dirs["l8"], base_name + ".tif")
                    
                    if not args.overwrite and os.path.exists(out_l8_path): continue

                    # 1. Extraction L8
                    win = Window(c, r, PATCH_SIZE, PATCH_SIZE)
                    l8_patch = src_l8.read(window=win)
                    
                    # Filtre NoData
                    if get_nodata_mask(l8_patch).mean() > NODATA_THR: continue

                    # 2. Extraction S2 (Mosaïque sur Bbox)
                    try:
                        pts = [trans * (c, r+PATCH_SIZE), trans * (c+PATCH_SIZE, r)] # BL, TR
                        bbox = [min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1]),
                                max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])]
                        
                        s2_patch, s2_trans = mosaic_s2(paths_s2, bbox, (S2_PATCH_SIZE, S2_PATCH_SIZE))
                    except Exception:
                        continue # Skip si S2 illisible/manquant

                    if get_nodata_mask(s2_patch).mean() > NODATA_THR: continue

                    # 3. Alignement
                    dy, dx = 0.0, 0.0
                    try:
                        s2_small = resize_chw(s2_patch, PATCH_SIZE, PATCH_SIZE)
                        dy_low, dx_low = estimate_shift(l8_patch, s2_small, upsample=ALIGN_UPSAMPLE)
                        
                        # Rejet si shift aberrant (>8px)
                        if abs(dy_low) < 8 and abs(dx_low) < 8:
                            s2_patch = apply_shift(s2_patch, dy_low * S2_SCALE, dx_low * S2_SCALE)
                            dy, dx = dy_low, dx_low
                    except Exception as e:
                        logger.debug(f"Align failed {base_name}: {e}")

                    # 4. Métriques et Sauvegarde
                    metrics = compute_metrics(l8_patch, s2_patch)
                    score = metrics["ssim"]

                    if args.visu:
                        save_debug_viz(l8_patch, s2_patch, os.path.join(dirs["vis"], base_name + ".png"),
                                       f"Score: {score:.3f} | Shift: {dy:.2f}, {dx:.2f}")

                    if score >= args.threshold:
                        # Write L8
                        meta.update({"height": PATCH_SIZE, "width": PATCH_SIZE, 
                                     "transform": rasterio.windows.transform(win, trans)})
                        with rasterio.open(out_l8_path, "w", **meta) as dst:
                            dst.write(l8_patch)
                        
                        # Write S2
                        out_s2_path = os.path.join(dirs["s2"], base_name + ".tif")
                        meta_s2 = meta.copy()
                        meta_s2.update({"height": S2_PATCH_SIZE, "width": S2_PATCH_SIZE, 
                                        "transform": s2_trans, "dtype": "float32"})
                        with rasterio.open(out_s2_path, "w", **meta_s2) as dst:
                            dst.write(s2_patch)

                        catalogue.append({
                            "id": base_name, "score": score, "zncc": metrics["zncc"],
                            "l8_path": out_l8_path, "s2_path": out_s2_path
                        })

    # Export & Splitting
    df = pd.DataFrame(catalogue)
    if not df.empty and args.split > 0:
        logger.info(f"Splitting dataset (Ratio: {args.split})...")
        export2 = args.export + "_val"
        os.makedirs(os.path.join(export2, "landsat8"), exist_ok=True)
        os.makedirs(os.path.join(export2, "sentinel2"), exist_ok=True)
        
        val_df = df.sample(frac=args.split, random_state=42)
        train_df = df.drop(val_df.index)
        
        for _, row in val_df.iterrows():
            shutil.move(row["l8_path"], os.path.join(export2, "landsat8", os.path.basename(row["l8_path"])))
            shutil.move(row["s2_path"], os.path.join(export2, "sentinel2", os.path.basename(row["s2_path"])))
        
        train_df.to_csv(os.path.join(args.export, "catalogue_train.csv"), index=False)
        val_df.to_csv(os.path.join(export2, "catalogue_val.csv"), index=False)
    else:
        if not df.empty: df.to_csv(os.path.join(args.export, "catalogue.csv"), index=False)

    logger.info("Traitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracteur de patchs L8/S2 Optimisé")
    parser.add_argument('--landsat', required=True, help="Dossier Landsat 8")
    parser.add_argument('--sentinel', required=True, help="Dossier Sentinel 2")
    parser.add_argument('--export', default="dataset_output", help="Dossier de sortie")
    parser.add_argument('--threshold', type=float, default=0.60, help="Seuil de qualité SSIM (0.0-1.0)")
    parser.add_argument('--split', type=float, default=0.20, help="Ratio pour le set de validation (ex: 0.2)")
    parser.add_argument('--visu', action='store_true', help="Générer des images de debug")
    parser.add_argument('--overwrite', action='store_true', help="Ecraser les données existantes")
    
    process_patches(parser.parse_args())