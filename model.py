"""
model.py — PoseCNN for Animal Kingdom pose estimation
======================================================
Architecture:
    Encoder : ResNet-18 pretrained on ImageNet (strips avg-pool + FC)
              Input  (B, 3, H, W)  ->  features (B, 512, H/32, W/32)

    Head    : 3 x ConvTranspose2d upsampling blocks
              (B, 512, H/32, W/32) -> heatmaps (B, K, H/4, W/4)

    Decode  : soft_argmax converts heatmaps to differentiable (x,y) coords
              (B, K, H/4, W/4)    -> coords (B, K, 2)  in normalised [0,1]

Keypoint index map (Animal Kingdom MPII-style, 23 joints):
     0  - right ankle
     1  - right knee
     2  - right hip
     3  - left hip
     4  - left knee
     5  - left ankle
     6  - right wrist
     7  - right elbow
     8  - right shoulder
     9  - left shoulder
    10  - left elbow
    11  - left wrist
    12  - neck / throat
    13  - head top
    14  - tail base
    15  - tail mid
    16  - tail tip
    17  - front-right paw
    18  - front-left paw
    19  - back-right paw
    20  - back-left paw
    21  - snout / nose
    22  - left ear  (or spare landmark)

Note: exact semantic meaning varies by species — visibility flags are
the ground truth signal for which joints are meaningful per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ── Animal Kingdom skeleton ───────────────────────────────────────────────────
# Each tuple is (parent_idx, child_idx) — defines a bone for loss_bone.
# Built to be reasonable across quadrupeds, birds, fish, reptiles etc.
BONES = [
    (0,  1),   # right ankle  -> right knee
    (1,  2),   # right knee   -> right hip
    (2,  3),   # right hip    -> left hip      (spine base)
    (3,  4),   # left hip     -> left knee
    (4,  5),   # left knee    -> left ankle
    (6,  7),   # right wrist  -> right elbow
    (7,  8),   # right elbow  -> right shoulder
    (8,  9),   # right shoulder -> left shoulder  (spine top)
    (9, 10),   # left shoulder -> left elbow
    (10, 11),  # left elbow   -> left wrist
    (8, 12),   # right shoulder -> neck
    (9, 12),   # left shoulder  -> neck
    (12, 13),  # neck          -> head top
    (12, 21),  # neck          -> snout
    (2, 14),   # right hip     -> tail base
    (3, 14),   # left hip      -> tail base
    (14, 15),  # tail base     -> tail mid
    (15, 16),  # tail mid      -> tail tip
    (8, 17),   # right shoulder -> front-right paw
    (9, 18),   # left shoulder  -> front-left paw
    (2, 19),   # right hip      -> back-right paw
    (3, 20),   # left hip       -> back-left paw
    (13, 22),  # head top       -> left ear
]

# Joint triplets for loss_angle: (a, vertex, b) — angle is at vertex
ANGLE_TRIPLETS = [
    (0,  1,  2),   # ankle-knee-hip        (right)
    (5,  4,  3),   # ankle-knee-hip        (left)
    (6,  7,  8),   # wrist-elbow-shoulder  (right)
    (11, 10, 9),   # wrist-elbow-shoulder  (left)
    (2,  8, 12),   # hip-shoulder-neck     (right)
    (3,  9, 12),   # hip-shoulder-neck     (left)
    (8, 12, 13),   # shoulder-neck-head
    (12, 13, 21),  # neck-head-snout
    (2,  14, 15),  # hip-tailbase-tailmid
    (14, 15, 16),  # tailbase-tailmid-tailtip
]


# ── Soft-argmax ───────────────────────────────────────────────────────────────
def soft_argmax(heatmaps, temperature=10.0):
    """
    Convert heatmaps to differentiable (x, y) coordinates.

    Parameters
    ----------
    heatmaps    : (B, K, H, W)  — raw heatmap logits
    temperature : float         — sharpens the softmax peak

    Returns
    -------
    coords : (B, K, 2)  — (x, y) in normalised [0, 1]
    """
    B, K, H, W = heatmaps.shape

    flat    = heatmaps.reshape(B, K, -1)
    weights = F.softmax(flat * temperature, dim=-1)

    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)

    grid_x = xs.unsqueeze(0).expand(H, -1).reshape(-1)
    grid_y = ys.unsqueeze(1).expand(-1, W).reshape(-1)

    pred_x = (weights * grid_x).sum(-1)
    pred_y = (weights * grid_y).sum(-1)

    return torch.stack([pred_x, pred_y], dim=-1)   # (B, K, 2)


# ── Upsampling block ──────────────────────────────────────────────────────────
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── PoseCNN ───────────────────────────────────────────────────────────────────
class PoseCNN(nn.Module):
    """
    Lightweight pose estimation network.

    Encoder: ResNet-18 pretrained backbone.
    Head:    3 transposed convolution blocks + final 1x1 conv.

    For a 256x256 input:
        Encoder output : (B, 512,  8,  8)
        After block 1  : (B, 256, 16, 16)
        After block 2  : (B, 128, 32, 32)
        After block 3  : (B,  64, 64, 64)
        Heatmap output : (B,   K, 64, 64)
    """

    def __init__(self, num_keypoints=23, pretrained=True, freeze_encoder=False):
        """
        Parameters
        ----------
        num_keypoints  : int   — 23 for Animal Kingdom
        pretrained     : bool  — ImageNet pretrained ResNet-18
        freeze_encoder : bool  — freeze backbone for warm-up epochs
        """
        super().__init__()

        weights  = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.up1 = UpsampleBlock(512, 256)
        self.up2 = UpsampleBlock(256, 128)
        self.up3 = UpsampleBlock(128,  64)
        self.heatmap_conv = nn.Conv2d(64, num_keypoints, kernel_size=1)

        self._init_head()

    def _init_head(self):
        for module in [self.up1, self.up2, self.up3]:
            for m in module.modules():
                if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias,   0)
        nn.init.normal_(self.heatmap_conv.weight, std=0.01)
        nn.init.constant_(self.heatmap_conv.bias, 0)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("[model] Encoder unfrozen — full network is now trainable.")

    def forward(self, x):
        features = self.encoder(x)
        x = self.up1(features)
        x = self.up2(x)
        x = self.up3(x)
        return self.heatmap_conv(x)      # (B, K, H/4, W/4)

    def predict_coords(self, x, temperature=10.0):
        heatmaps = self.forward(x)
        coords   = soft_argmax(heatmaps, temperature=temperature)
        return coords, heatmaps

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from dataset import NUM_KEYPOINTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print("=" * 60)

    model = PoseCNN(num_keypoints=NUM_KEYPOINTS, pretrained=True).to(device)
    total, trainable = model.count_parameters()
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Bones defined        : {len(BONES)}")
    print(f"Angle triplets       : {len(ANGLE_TRIPLETS)}")

    # Single frame
    x      = torch.randn(1, 3, 256, 256, device=device)
    hm     = model(x)
    coords = soft_argmax(hm)
    print(f"\nInput  : {x.shape}")
    print(f"Heatmap: {hm.shape}")        # (1, 23, 64, 64)
    print(f"Coords : {coords.shape}")    # (1, 23, 2)
    print(f"Range  : [{coords.min():.3f}, {coords.max():.3f}]")

    # Triplet batch
    B = 4
    all_frames = torch.randn(B * 3, 3, 256, 256, device=device)
    all_coords = soft_argmax(model(all_frames))
    c0, c1, c2 = all_coords.chunk(3, dim=0)

    loss = c1.mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print(f"\nTriplet coords : {c0.shape} x3")
    print(f"Gradients flow : {grad_ok}")
    print("\nmodel.py checks passed ✓  →  ready for losses.py")