#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6_Applique_ESRGAN.py

Application du modèle ESRGAN (Générateur) sur les images de Test (20%).
Génère des images Super-Resolution avec détails fins.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    sys.exit("Installez rasterio : pip install rasterio")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# --- ARCHITECTURE (DOIT MATCHER SCRIPT 5) ---

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


# --- UTILS TILING ---

def generate_grid(H, W, tile, overlap):
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    for y in ys:
        for x in xs: yield y, y+tile, x, x+tile

def process_file(src_path, dst_path, model, device, args):
    with rasterio.open(src_path) as src:
        # Lecture
        bands = [int(b)+1 for b in args.bands.split(",")]
        data = src.read(bands)
        meta = src.meta.copy()
        tr = src.transform

    # Nettoyage & Scale
    data = np.nan_to_num(data, nan=0.0)
    C, H, W = data.shape
    
    # Init Buffers
    scale = args.scale
    H_out, W_out = H*scale, W*scale
    out_sum = np.zeros((C, H_out, W_out), dtype=np.float32)
    w_sum = np.zeros((1, H_out, W_out), dtype=np.float32)
    
    # Tile size HR
    tile_hr = args.tile * scale
    
    # Inférence
    with torch.no_grad():
        grid = list(generate_grid(H, W, args.tile, args.overlap))
        for y0, y1, x0, x1 in tqdm(grid, desc=src_path.name, leave=False):
            # Crop LR
            patch = data[:, y0:y1, x0:x1]
            
            # Pad si bord
            ph = args.tile - patch.shape[1]
            pw = args.tile - patch.shape[2]
            if ph>0 or pw>0: patch = np.pad(patch, ((0,0),(0,ph),(0,pw)), 'reflect')
            
            # Tensor
            t_in = torch.from_numpy(patch).unsqueeze(0).to(device).float()
            t_in /= 10000.0 # Normalisation
            
            # Forward
            with torch.amp.autocast('cuda'):
                t_out = model(t_in)
            
            # Back to Numpy
            sr = t_out.squeeze(0).cpu().numpy()
            sr = np.clip(sr, 0, 1.5) * 10000.0
            
            # Crop valid & Add
            vh, vw = (y1-y0)*scale, (x1-x0)*scale
            sr = sr[:, :vh, :vw]
            
            y0h, x0h = y0*scale, x0*scale
            out_sum[:, y0h:y0h+vh, x0h:x0h+vw] += sr
            w_sum[:, y0h:y0h+vh, x0h:x0h+vw] += 1.0

    # Average
    final = out_sum / np.maximum(w_sum, 1e-6)
    
    # Meta Update
    new_tr = Affine(tr.a/scale, tr.b, tr.c, tr.d, tr.e/scale, tr.f)
    meta.update({"height": H_out, "width": W_out, "transform": new_tr, "dtype": "float32"})
    
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(final)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--bands", default="2,3,4,5") # Indices 0-based pour rasterio (2=Blue, 3=Green, 4=Red, 5=NIR)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    dst = Path(args.dst_dir); dst.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    # Load Model
    net = Generator(n_feats=128, n_blocks=16, scale=args.scale).to(dev)
    print(f"Load Checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=dev)
    net.load_state_dict(state, strict=True)
    net.eval()

    # Process
    files = list(Path(args.src_dir).glob("*.tif"))
    print(f"Traitement de {len(files)} fichiers...")
    
    for f in files:
        out_name = f.name.replace("L8", "ESRGAN").replace("LR", "ESRGAN")
        if out_name == f.name: out_name = "ESRGAN_" + f.name
        
        try:
            process_file(f, dst / out_name, net, dev, args)
            print(f"OK: {out_name}")
        except Exception as e:
            print(f"ERR {f.name}: {e}")

if __name__ == "__main__":
    main()