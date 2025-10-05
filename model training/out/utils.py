# utils.py
import math, os, cv2, torch, numpy as np
from typing import Tuple, List, Optional
from skimage.morphology import disk
from skimage.measure import label, regionprops
from PIL import Image

def load_gray_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)

def save_png(path: str, arr: np.ndarray):
    Image.fromarray(arr).save(path)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def points_csv_to_mask(csv_path: str, H: int, W: int, radius: int = 3) -> np.ndarray:
    """
    Turn sparse positive pixels (x,y per line) into a small-disk-dilated binary mask.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(csv_path):
        return mask
    pts = []
    with open(csv_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if "," in line:
                x,y = line.split(",")[:2]
            else:
                x,y = line.split()[:2]
            x=int(float(x)); y=int(float(y))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1
    if mask.sum()==0:
        return mask
    d = disk(radius).astype(np.uint8)
    # Dilate via convolution
    kh, kw = d.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(mask, ((pad_h,pad_h),(pad_w,pad_w)), mode="constant")
    out = np.zeros_like(mask)
    for i in range(H):
        for j in range(W):
            if (padded[i:i+kh, j:j+kw]*d).sum()>0:
                out[i,j]=1
    return out

def make_training_tiles(img: np.ndarray, mask: np.ndarray, tile: int=256, stride: int=256,
                        pos_frac: float=0.5, max_tiles:int=4000, pos_thresh:int=20):
    """
    Sample tiles with class-balance: ~pos_frac contain positives.
    """
    H, W = img.shape
    tiles = []
    pos_tiles, neg_tiles = [], []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            m = mask[y:y+tile, x:x+tile]
            portion = int(m.sum())
            if portion >= pos_thresh:
                pos_tiles.append((x,y))
            else:
                neg_tiles.append((x,y))
    # If nothing labelled yet, just sample grid
    if len(pos_tiles)==0 and len(neg_tiles)==0:
        for y in range(0, H - tile + 1, stride):
            for x in range(0, W - tile + 1, stride):
                tiles.append((x,y))
        return tiles[:max_tiles]

    n_pos = int(max_tiles * pos_frac)
    n_neg = max_tiles - n_pos
    rng = np.random.default_rng(42)
    rng.shuffle(pos_tiles); rng.shuffle(neg_tiles)
    tiles = pos_tiles[:n_pos] + neg_tiles[:n_neg]
    rng.shuffle(tiles)
    return tiles

def normalize(img: np.ndarray) -> np.ndarray:
    # simple 0-1 normalize per-image
    arr = img.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = arr*0
    return arr

def overlay_red(base_gray: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    base_gray: uint8 HxW
    pred_mask: bool/0-1 HxW
    Returns RGB uint8 with red overlay.
    """
    base = base_gray
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base.copy()
    overlay = rgb.copy()
    overlay[pred_mask > 0] = np.array([255,0,0], dtype=np.uint8)
    return overlay

def count_ships(binary_mask: np.ndarray, min_area:int=15, max_area:int=4000,
                min_aspect:float=1.5, # ships more elongated (major/minor)
                ) -> Tuple[int, List[Tuple[int,int,int,int]]]:
    """
    Count ships via connected components + simple geometry filters.
    Returns (count, list_of_bboxes)
    """
    lab = label(binary_mask.astype(np.uint8), connectivity=2)
    props = regionprops(lab)
    bboxes=[]
    for p in props:
        a = p.area
        if a < min_area or a > max_area:
            continue
        # elongation via major/minor axis (fallback if undefined)
        maj = getattr(p, "major_axis_length", 1.0) or 1.0
        minr = getattr(p, "minor_axis_length", 1.0) or 1.0
        aspect = maj / max(minr,1e-6)
        if aspect < min_aspect:
            continue
        y0, x0, y1, x1 = p.bbox
        bboxes.append((x0,y0,x1,y1))
    return len(bboxes), bboxes

def stitch_probs(prob_tiles, H, W, tile, stride, device="cpu"):
    """
    Blend overlapping tiles by averaging.
    """
    prob_map = np.zeros((H,W), dtype=np.float32)
    wei_map  = np.zeros((H,W), dtype=np.float32)
    idx=0
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            p = prob_tiles[idx]
            prob_map[y:y+tile, x:x+tile] += p
            wei_map[y:y+tile, x:x+tile]  += 1.0
            idx+=1
    wei_map[wei_map==0]=1.0
    prob_map /= wei_map
    return prob_map
