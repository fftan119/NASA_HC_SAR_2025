# infer.py
import os, argparse, torch, numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from utils import (ensure_dir, load_gray_png, save_png,
                   overlay_red, stitch_probs, count_ships, normalize)
from train import UNetSmall

def sliding_predict(img: np.ndarray, ckpt_path: str, tile:int=512, stride:int=384, thresh:float=0.5):
    """
    img: uint8 HxW grayscale
    returns prob_map [0..1], bin_mask 0/1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNetSmall(in_ch=1, ch=32).to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["model"])
    net.eval()

    H,W = img.shape
    tiles_probs=[]

    with torch.no_grad():
        for y in tqdm(range(0, H - tile + 1, stride), desc="Tiles (y)"):
            for x in range(0, W - tile + 1, stride):
                patch = img[y:y+tile, x:x+tile]
                patch = normalize(patch)
                t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    logits = net(t)
                    probs = torch.sigmoid(logits).squeeze().float().cpu().numpy()
                tiles_probs.append(probs)
    prob_map = stitch_probs(tiles_probs, H, W, tile, stride)
    bin_mask = (prob_map >= thresh).astype(np.uint8)
    return prob_map, bin_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input PNG")
    ap.add_argument("--ckpt",  required=True, help="path to best.pt")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument("--min_area", type=int, default=15)
    ap.add_argument("--max_area", type=int, default=4000)
    ap.add_argument("--min_aspect", type=float, default=1.4)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    img = load_gray_png(args.image)
    H,W = img.shape

    prob, mask = sliding_predict(img, ckpt_path=args.ckpt,
                                 tile=args.tile, stride=args.stride, thresh=args.thresh)

    # Post-process: remove tiny noise by simple opening/closing (cv2)
    import cv2
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(mask*255, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean = (clean>0).astype(np.uint8)

    count, bboxes = count_ships(clean, min_area=args.min_area, max_area=args.max_area,
                                min_aspect=args.min_aspect)
    overlay = overlay_red(img, clean)

    base = os.path.splitext(os.path.basename(args.image))[0]
    save_png(os.path.join(args.out_dir, f"{base}_overlay.png"), overlay)
    # Also save a probability heatmap for debugging (optional)
    (Image.fromarray((prob*255).astype(np.uint8))
        .save(os.path.join(args.out_dir, f"{base}_prob.png")))

    print(f"Detected ships: {count}")
    # Save a simple text report with bboxes
    with open(os.path.join(args.out_dir, f"{base}_report.txt"), "w") as f:
        f.write(f"ships={count}\n")
        for (x0,y0,x1,y1) in bboxes:
            f.write(f"{x0},{y0},{x1},{y1}\n")

if __name__ == "__main__":
    main()
