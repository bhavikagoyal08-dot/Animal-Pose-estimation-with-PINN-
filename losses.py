"""
losses.py — PINN Loss Functions for Animal Kingdom Pose Estimation
==================================================================
Three physics-informed loss terms:

    L_data   : supervised MSE on annotated keypoints (visibility-masked)
    L_bone   : rigid body prior — penalises deviation from reference bone lengths
    L_smooth : temporal smoothness — penalises high acceleration (2nd finite diff)

Optional:
    L_angle  : soft joint angle constraints

Combined:
    L_total = L_data + lam_bone * L_bone + lam_smooth * L_smooth
            [+ lam_angle * L_angle  if joint_triples provided]

Recommended starting weights:
    lam_bone   = 0.5
    lam_smooth = 0.1
    lam_angle  = 0.05
"""

import torch
import torch.nn.functional as F


# ── Loss 1: Data loss ─────────────────────────────────────────────────────────
def loss_data(pred_coords, gt_coords, visibility):
    """
    Visibility-masked MSE between predicted and ground-truth keypoints.

    Only penalises joints that are annotated (visibility=1).
    Invisible joints (visibility=0) are excluded — their coords are
    stored as [0,0] in Animal Kingdom after masking, so including them
    would corrupt the gradient.

    Parameters
    ----------
    pred_coords : (B, K, 2)  — predicted normalised (x,y) from soft_argmax
    gt_coords   : (B, K, 2)  — ground truth normalised (x,y) from dataset
    visibility  : (B, K)     — 1.0 if labelled, 0.0 if occluded/missing

    Returns
    -------
    scalar tensor
    """
    diff = (pred_coords - gt_coords) ** 2           # (B, K, 2)
    diff = diff.sum(dim=-1)                          # (B, K)
    loss = (diff * visibility).sum() / (visibility.sum() + 1e-6)
    return loss


# ── Loss 2: Bone length loss ──────────────────────────────────────────────────
def loss_bone(pred_coords, bone_lengths_ref):
    """
    Penalises deviation from reference bone lengths.

    bone_lengths_ref is estimated once from the training set before
    training begins (see estimate_bone_lengths below).

    Parameters
    ----------
    pred_coords      : (B, K, 2)           — predicted normalised coords
    bone_lengths_ref : dict {(i,j): float} — reference lengths in [0,1] space

    Returns
    -------
    scalar tensor
    """
    if not bone_lengths_ref:
        return torch.tensor(0.0, device=pred_coords.device)

    total = torch.tensor(0.0, device=pred_coords.device)
    for (i, j), ref_len in bone_lengths_ref.items():
        seg  = pred_coords[:, i, :] - pred_coords[:, j, :]
        dist = seg.norm(dim=-1)
        ref  = torch.tensor(ref_len, device=pred_coords.device,
                            dtype=torch.float32)
        total = total + ((dist - ref) ** 2).mean()

    return total / len(bone_lengths_ref)


# ── Loss 3: Temporal smoothness loss ──────────────────────────────────────────
def loss_smooth(coords_t0, coords_t1, coords_t2):
    """
    Penalises high acceleration via the second finite difference.

        accel = coords_t2 - 2 * coords_t1 + coords_t0

    Parameters
    ----------
    coords_t0, coords_t1, coords_t2 : each (B, K, 2)

    Returns
    -------
    scalar tensor
    """
    accel = coords_t2 - 2.0 * coords_t1 + coords_t0
    return (accel ** 2).mean()


# ── Loss 4 (optional): Joint angle loss ───────────────────────────────────────
def loss_angle(pred_coords, joint_triples, min_deg=20.0, max_deg=160.0):
    """
    Soft hinge constraint on joint angles.

    Zero gradient when angle is within [min_deg, max_deg].
    Quadratic penalty outside that range.

    Parameters
    ----------
    pred_coords   : (B, K, 2)
    joint_triples : list of (a, vertex, c)
    min_deg       : float
    max_deg       : float

    Returns
    -------
    scalar tensor
    """
    if not joint_triples:
        return torch.tensor(0.0, device=pred_coords.device)

    min_r = torch.tensor(min_deg * 3.14159265 / 180.0,
                         device=pred_coords.device)
    max_r = torch.tensor(max_deg * 3.14159265 / 180.0,
                         device=pred_coords.device)

    total = torch.tensor(0.0, device=pred_coords.device)
    for (a, b, c) in joint_triples:
        va = pred_coords[:, a, :] - pred_coords[:, b, :]
        vc = pred_coords[:, c, :] - pred_coords[:, b, :]

        cos_theta = F.cosine_similarity(va, vc, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta     = torch.acos(cos_theta)

        below = F.relu(min_r - theta)
        above = F.relu(theta - max_r)
        total = total + (below ** 2 + above ** 2).mean()

    return total / len(joint_triples)


# ── Combined PINN loss ────────────────────────────────────────────────────────
def pinn_loss(coords_t0, coords_t1, coords_t2,
              gt_coords, visibility,
              bone_lengths_ref,
              joint_triples=None,
              lam_bone=0.5,
              lam_smooth=0.1,
              lam_angle=0.05):
    """
    Combined PINN training loss.

    gt_coords and visibility correspond to the MIDDLE frame (t1).

    Parameters
    ----------
    coords_t0, coords_t1, coords_t2 : each (B, K, 2)
    gt_coords        : (B, K, 2)
    visibility       : (B, K)
    bone_lengths_ref : dict {(i,j): float}
    joint_triples    : list of (a,b,c) or None
    lam_bone         : float
    lam_smooth       : float
    lam_angle        : float

    Returns
    -------
    total : scalar tensor
    log   : dict of individual loss values
    """
    l_data   = loss_data(coords_t1, gt_coords, visibility)
    l_bone   = loss_bone(coords_t1, bone_lengths_ref)
    l_smooth = loss_smooth(coords_t0, coords_t1, coords_t2)

    total = l_data + lam_bone * l_bone + lam_smooth * l_smooth
    log   = {
        "loss_total":  total.item(),
        "loss_data":   l_data.item(),
        "loss_bone":   l_bone.item(),
        "loss_smooth": l_smooth.item(),
    }

    if joint_triples:
        l_angle = loss_angle(coords_t1, joint_triples)
        total   = total + lam_angle * l_angle
        log["loss_angle"] = l_angle.item()
        log["loss_total"] = total.item()

    return total, log


# ── Bone length estimator ─────────────────────────────────────────────────────
def estimate_bone_lengths(entries, bones):
    """
    Estimate reference bone lengths by averaging over all training entries
    where BOTH endpoints of a bone are visible.

    Call this once before training and pass the result to pinn_loss.

    Parameters
    ----------
    entries : list of entry dicts from load_all_entries()
    bones   : list of (i, j) index pairs — e.g. BONES from model.py

    Returns
    -------
    dict {(i, j): float}  — mean bone length in normalised [0,1] coords
    """
    import numpy as np
    from dataset import IMG_W, IMG_H

    accum = {bone: [] for bone in bones}

    for entry in entries:
        joints     = np.array(entry["joints"],     dtype=np.float32)   # (23, 2)
        joints_vis = np.array(entry["joints_vis"], dtype=np.float32)   # (23,)

        # Mask invisible
        invisible = (joints[:, 0] < 0) | (joints[:, 1] < 0)
        joints_vis[invisible] = 0.0
        joints[invisible]     = 0.0

        # Normalise
        joints[:, 0] /= IMG_W
        joints[:, 1] /= IMG_H

        for (i, j) in bones:
            if joints_vis[i] > 0 and joints_vis[j] > 0:
                length = np.linalg.norm(joints[i] - joints[j])
                accum[(i, j)].append(length)

    bone_lengths = {}
    for (i, j), lengths in accum.items():
        if lengths:
            bone_lengths[(i, j)] = float(np.mean(lengths))

    print(f"  Bone lengths estimated: {len(bone_lengths)}/{len(bones)} bones "
          f"had visible data")
    return bone_lengths


# ── PCK metric ────────────────────────────────────────────────────────────────
def pck_metric(pred_coords, gt_coords, visibility, threshold=0.05):
    """
    Percentage of Correct Keypoints (PCK).

    threshold=0.05 → 5% of image diagonal in normalised coords.

    Parameters
    ----------
    pred_coords : (B, K, 2)
    gt_coords   : (B, K, 2)
    visibility  : (B, K)
    threshold   : float

    Returns
    -------
    pck : float in [0, 100]
    """
    with torch.no_grad():
        dist    = (pred_coords - gt_coords).norm(dim=-1)
        correct = (dist < threshold).float()
        n_vis   = visibility.sum()
        if n_vis < 1:
            return 0.0
        pck = (correct * visibility).sum() / n_vis
    return pck.item() * 100.0


# ── Jitter metric ─────────────────────────────────────────────────────────────
def jitter_metric(coords_t0, coords_t1):
    """
    Mean frame-to-frame displacement of predicted keypoints.
    Lower = smoother trajectory.
    """
    with torch.no_grad():
        disp = (coords_t1 - coords_t0).norm(dim=-1)
    return disp.mean().item()


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import AnimalKingdomTripletDataset, NUM_KEYPOINTS
    from model import PoseCNN, soft_argmax, BONES, ANGLE_TRIPLETS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load a small batch
    print("\nLoading dataset (train split)...")
    ds     = AnimalKingdomTripletDataset(split="train", img_size=(256, 256))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch  = next(iter(loader))

    t0         = batch["img_t0"].to(device)
    t1         = batch["img_t1"].to(device)
    t2         = batch["img_t2"].to(device)
    gt_coords  = batch["kp_t1"].to(device)
    visibility = batch["vis_t1"].to(device)

    # Estimate bone lengths from a small subset (first 500 entries for speed)
    print("\nEstimating bone lengths...")
    bone_lengths = estimate_bone_lengths(ds.entries[:500], BONES)

    # Build model and get predictions
    model = PoseCNN(num_keypoints=NUM_KEYPOINTS).to(device)
    model.eval()
    with torch.no_grad():
        all_frames = torch.cat([t0, t1, t2], dim=0)
        all_coords = soft_argmax(model(all_frames))
    coords_t0, coords_t1, coords_t2 = all_coords.chunk(3, dim=0)

    # Enable grad for loss test
    coords_t0 = coords_t0.detach().requires_grad_(True)
    coords_t1 = coords_t1.detach().requires_grad_(True)
    coords_t2 = coords_t2.detach().requires_grad_(True)

    print("\n" + "=" * 60)
    print("Testing individual losses...")
    l_d = loss_data(coords_t1, gt_coords, visibility)
    l_b = loss_bone(coords_t1, bone_lengths)
    l_s = loss_smooth(coords_t0, coords_t1, coords_t2)
    l_a = loss_angle(coords_t1, ANGLE_TRIPLETS)
    print(f"  L_data   : {l_d.item():.6f}")
    print(f"  L_bone   : {l_b.item():.6f}")
    print(f"  L_smooth : {l_s.item():.6f}")
    print(f"  L_angle  : {l_a.item():.6f}")

    print("\n" + "=" * 60)
    print("Testing combined pinn_loss...")
    total, log = pinn_loss(
        coords_t0, coords_t1, coords_t2,
        gt_coords, visibility,
        bone_lengths,
        joint_triples=ANGLE_TRIPLETS,
        lam_bone=0.5, lam_smooth=0.1, lam_angle=0.05,
    )
    for k, v in log.items():
        print(f"  {k:20s}: {v:.6f}")

    print("\n" + "=" * 60)
    print("Testing gradient flow through combined loss...")
    model.train()
    all_coords = soft_argmax(model(torch.cat([t0, t1, t2], dim=0)))
    c0, c1, c2 = all_coords.chunk(3, dim=0)
    total, log = pinn_loss(c0, c1, c2, gt_coords, visibility, bone_lengths,
                           joint_triples=ANGLE_TRIPLETS)
    total.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print(f"  Gradients flow : {grad_ok}")
    print(f"  Combined loss  : {log['loss_total']:.6f}")

    print("\n" + "=" * 60)
    print("Testing metrics...")
    pck    = pck_metric(c1.detach(), gt_coords, visibility, threshold=0.05)
    jitter = jitter_metric(c0.detach(), c1.detach())
    print(f"  PCK @ 5%  : {pck:.2f}%  (expect low — untrained model)")
    print(f"  Jitter    : {jitter:.5f}")

    print("\nlosses.py checks passed ✓  →  ready for train.py")