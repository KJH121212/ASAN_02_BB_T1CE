# ================================================================
# datasets/bb2t1ce_dataset.py
# BB → T1CE paired dataset loader (2D version using MONAI)
# ================================================================

import os
import torch
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ToTensord,
)

def collapse_z_to_batch(tensor):
    """(B, C, Z, H, W) → (B*Z, C, H, W)"""
    B, C, Z, H, W = tensor.shape
    return tensor.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)


def get_bb2t1ce_dataloader(
    bb_dir,
    t1ce_dir,
    batch_size=1,
    num_workers=4,
    resample_spacing=(1.0, 1.0, 1.0),
    as_2d=True,
):
    """Return a MONAI DataLoader for BB–T1CE paired MRI volumes (2D)."""

    # ------------------------------------------------------------
    # (1) 파일 매칭
    # ------------------------------------------------------------
    bb_files = sorted([f for f in glob(os.path.join(bb_dir, "*.nii*")) if "@SynoEAStream" not in f])
    t1ce_files = sorted([f for f in glob(os.path.join(t1ce_dir, "*.nii*")) if "@SynoEAStream" not in f])

    assert len(bb_files) == len(t1ce_files), "❌ BB/T1CE pair count mismatch"
    data_dicts = [{"bb": b, "t1ce": t} for b, t in zip(bb_files, t1ce_files)]
    print(f"[INFO] Found {len(data_dicts)} BB–T1CE pairs.")

    # ------------------------------------------------------------
    # (2) Transform 구성 (3D로 읽고 intensity normalization만 수행)
    # ------------------------------------------------------------
    transforms = Compose([
        LoadImaged(keys=["bb", "t1ce"]),
        EnsureChannelFirstd(keys=["bb", "t1ce"]),
        Orientationd(keys=["bb", "t1ce"], axcodes="RAS"),
        Spacingd(keys=["bb", "t1ce"], pixdim=resample_spacing, mode=("bilinear", "bilinear")),
        ScaleIntensityRanged(
            keys=["bb", "t1ce"],
            a_min=0, a_max=4000,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=["bb", "t1ce"]),
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)

    # ------------------------------------------------------------
    # (3) Collate_fn: 3D → 2D 슬라이스 변환
    # ------------------------------------------------------------
    def collate_fn(batch):
        bb = torch.stack([b["bb"] for b in batch])     # (B, 1, Z, H, W)
        t1ce = torch.stack([b["t1ce"] for b in batch]) # (B, 1, Z, H, W)
        bb = collapse_z_to_batch(bb)                   # (B*Z, 1, H, W)
        t1ce = collapse_z_to_batch(t1ce)               # (B*Z, 1, H, W)
        return {"bb": bb, "t1ce": t1ce}


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader
