"""Test PatchCore on MVTEC-style carpet-like texture data."""

import json
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore


def test_patchcore(
    checkpoint_path: str = "outputs/patchcore/patchcore_final.ckpt", # path to trained model checkpoint
    data_dir: str = "data",
    output_dir: str = "outputs/patchcore_results",
):
    """run anomalib Engine.test and save resulting metrics"""

    data_root = Path(data_dir)
    ckpt = Path(checkpoint_path)
    out_root = Path(output_dir)

    if not ckpt.exists():
        print(f"Error: checkpoint not found at {ckpt}")
        return None

    abnormal_dirs: Sequence[str] = [f"test/{p.name}" for p in (data_root / "test").iterdir() if p.is_dir() and p.name != "good"]

    print("=" * 70)
    print("PatchCore Testing & Evaluation")
    print("=" * 70)
    print(f"Loading checkpoint: {ckpt}")

    datamodule = Folder(
        name="data_texture",
        root=data_root,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        train_batch_size=1, 
        eval_batch_size=1,
        num_workers=0,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",  
        layers=("layer2", "layer3"), # these should match the training config
        coreset_sampling_ratio=0.1, # this should match the training config
        num_neighbors=9,
    )

    trainer = pl.Trainer(
        default_root_dir=out_root, # directory to save logs and checkpoints
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
    )
    metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt), weights_only=False) # run test and get metrics with loaded checkpoint

    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest metrics:")
    print(metrics)
    print(f"Results saved: {results_path}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    test_patchcore(
        checkpoint_path="outputs/patchcore/patchcore_final.ckpt",
        data_dir="data",
        output_dir="outputs/patchcore_results",
    )
