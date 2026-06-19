import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ── IMAGE ROOTS (searched in order) ──────────────────────────────────────────
IMAGE_ROOTS = [
    Path(r"C:\Users\BHAVI\Downloads\dataset-002\dataset"),
    Path(r"C:\Users\BHAVI\Downloads\image-004\image"),
]

# ── ANNOTATION ROOT ───────────────────────────────────────────────────────────
ANNOTATION_ROOT = Path(
    r"C:\Users\BHAVI\Downloads\Animal_Kingdom-20260326T051203Z-1-001"
    r"\Animal_Kingdom\pose_estimation\annotation"
)

SUBFOLDERS = [
    "ak_P1",
    "ak_P2",
    "ak_P3_amphibian",
    "ak_P3_bird",
    "ak_P3_fish",
    "ak_P3_mammal",
    "ak_P3_reptile",
]

NUM_KEYPOINTS = 23
IMG_W, IMG_H  = 640, 360


# ── HELPERS ───────────────────────────────────────────────────────────────────

def find_image(image_field: str) -> Path:
    """Search both image roots for a clip frame."""
    for root in IMAGE_ROOTS:
        p = root / image_field
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Image not found in any root: {image_field}\n"
        f"  Searched: {[str(r) for r in IMAGE_ROOTS]}"
    )


def parse_clip_and_frame(image_field: str):
    """
    'AAJYPNPL/AAJYPNPL_f000011.jpg'  →  clip_id='AAJYPNPL', frame_num=11
    """
    stem = Path(image_field).stem              # 'AAJYPNPL_f000011'
    m = re.search(r'_f(\d+)$', stem)
    if m:
        clip_id   = stem[:m.start()]           # 'AAJYPNPL'
        frame_num = int(m.group(1))            # 11
    else:
        clip_id, frame_num = stem, 0
    return clip_id, frame_num


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_all_entries(split: str = "train") -> list:
    """
    Merge train.json / test.json from all 7 annotation subfolders
    into one flat list of entry dicts.
    """
    all_entries = []
    for folder in SUBFOLDERS:
        json_path = ANNOTATION_ROOT / folder / f"{split}.json"
        if not json_path.exists():
            print(f"  [WARN] Missing: {json_path}")
            continue
        with open(json_path, "r") as f:
            entries = json.load(f)
        print(f"  {folder}/{split}.json  →  {len(entries):>6} entries")
        all_entries.extend(entries)
    print(f"  {'─'*45}")
    print(f"  Total : {len(all_entries)} entries\n")
    return all_entries


def build_triplets(entries, min_clip_len= 3, max_triplets=None):
    """
    Group entries by clip ID, sort by frame number,
    emit all consecutive (t0, t1, t2) index triplets.
    """
    clip_map = defaultdict(list)   # clip_id → [(frame_num, entry_idx)]

    for idx, entry in enumerate(entries):
        clip_id, frame_num = parse_clip_and_frame(entry["image"])
        clip_map[clip_id].append((frame_num, idx))

    triplets = []
    skipped  = 0
    for clip_id, frame_list in clip_map.items():
        frame_list.sort(key=lambda x: x[0])
        indices = [idx for _, idx in frame_list]
        if len(indices) < min_clip_len:
            skipped += 1
            continue
        for i in range(len(indices) - 2):
            triplets.append((indices[i], indices[i + 1], indices[i + 2]))

    print(f"  Clips total     : {len(clip_map)}")
    print(f"  Clips skipped   : {skipped}  (fewer than {min_clip_len} frames)")
    print(f"  Triplets built  : {len(triplets)}\n")
    if max_triplets is not None:
        triplets = triplets[:max_triplets]
    return triplets


# ── PER-ENTRY LOADER ──────────────────────────────────────────────────────────

def entry_to_tensors(entry: dict, transform):
    """
    Load one entry → (img_tensor, keypoints, visibility)

    img_tensor  : (3, H, W)   float32, ImageNet-normalised
    keypoints   : (23, 2)     float32, normalised to [0, 1]
    visibility  : (23,)       float32, 1 = visible / 0 = invisible
    """
    # Image
    img_path   = find_image(entry["image"])
    img        = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    # Keypoints
    joints     = np.array(entry["joints"],     dtype=np.float32)   # (23, 2)
    joints_vis = np.array(entry["joints_vis"], dtype=np.float32)   # (23,)

    # Invisible joints are stored as [-1, -1] — zero them out
    invisible       = (joints[:, 0] < 0) | (joints[:, 1] < 0)
    joints_vis[invisible] = 0.0
    joints[invisible]     = 0.0

    # Normalise coordinates to [0, 1]
    joints[:, 0] /= IMG_W
    joints[:, 1] /= IMG_H

    keypoints  = torch.from_numpy(joints)        # (23, 2)
    visibility = torch.from_numpy(joints_vis)    # (23,)

    return img_tensor, keypoints, visibility


# ── DATASET ───────────────────────────────────────────────────────────────────

class AnimalKingdomTripletDataset(Dataset):
    """
    Temporal triplet dataset built from Animal Kingdom pose estimation
    annotations. Each item contains three consecutive frames (t0, t1, t2)
    from the same video clip.

    Args:
        split        : 'train' or 'test'
        img_size     : (H, W) images are resized to this before normalisation
        min_clip_len : clips shorter than this are excluded
    """

    def __init__(
        self,
        split: str         = "train",
        img_size: tuple    = (256, 256),
        min_clip_len: int  = 3,
        max_triplets=None,
    ):
        print(f"[AnimalKingdomTripletDataset]  split='{split}'")
        print(f"{'─' * 50}")

        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        self.entries  = load_all_entries(split)
        self.triplets = build_triplets(self.entries, min_clip_len, max_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        i0, i1, i2 = self.triplets[idx]

        img0, kp0, vis0 = entry_to_tensors(self.entries[i0], self.transform)
        img1, kp1, vis1 = entry_to_tensors(self.entries[i1], self.transform)
        img2, kp2, vis2 = entry_to_tensors(self.entries[i2], self.transform)

        clip_id, _ = parse_clip_and_frame(self.entries[i1]["image"])

        return {
            # ── Images ──────────────────────────────  (3, H, W)
            "img_t0":  img0,
            "img_t1":  img1,
            "img_t2":  img2,
            # ── Keypoints ───────────────────────────  (23, 2)  in [0,1]
            "kp_t0":   kp0,
            "kp_t1":   kp1,
            "kp_t2":   kp2,
            # ── Visibility flags ────────────────────  (23,)
            "vis_t0":  vis0,
            "vis_t1":  vis1,
            "vis_t2":  vis2,
            # ── Metadata ────────────────────────────
            "animal":  self.entries[i1]["animal"],
            "clip_id": clip_id,
        }


# ── QUICK SMOKE TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = AnimalKingdomTripletDataset(split="train", img_size=(256, 256))

    print(f"Dataset length : {len(dataset)}")
    sample = dataset[0]
    print(f"img_t1 shape   : {sample['img_t1'].shape}")   # (3, 256, 256)
    print(f"kp_t1  shape   : {sample['kp_t1'].shape}")    # (23, 2)
    print(f"vis_t1 shape   : {sample['vis_t1'].shape}")   # (23,)
    print(f"animal         : {sample['animal']}")
    print(f"clip_id        : {sample['clip_id']}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0
    )
    batch = next(iter(loader))
    print(f"\nBatch img_t1   : {batch['img_t1'].shape}")  # (4, 3, 256, 256)
    print(f"Batch kp_t1    : {batch['kp_t1'].shape}")     # (4, 23, 2)
    print("Smoke test passed ✓")