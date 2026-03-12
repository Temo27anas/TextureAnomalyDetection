# Train PatchCore model on the texture defect dataset and save the checkpoint

from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore


def train_patchcore(
    data_dir: str = "data", # path to dataset
    output_dir: str = "outputs/patchcore", # path to save model checkpoints and logs
    image_size: int = 256, # size to resize input images
    center_crop_size: int = 224, # size for center cropping after resizing
    backbone: str = "wide_resnet50_2", 
    layers: tuple = ("layer2", "layer3"), # layers from backbone to extract features
    coreset_sampling_ratio: float = 0.1, # ratio of features to sample for coreset
    num_neighbors: int = 9, # number of nearest neighbors for anomaly scoring
    max_epochs: int = 1, 
    batch_size: int = 8): 

    """train PatchCore model and return the resolved checkpoint path"""

    data_root = Path(data_dir)
    out_root = Path(output_dir)

    if not (data_root/"train"/"good").exists():
        raise FileNotFoundError(f"Missing training folder: {data_root / 'train' / 'good'}")

    defect_names: Sequence[str] = sorted(p.name for p in (data_root / "test").iterdir() if p.is_dir() and p.name != "good")
    abnormal_dirs: Sequence[str] = [f"test/{name}" for name in defect_names]

    print("=" * 70)
    print("PatchCore Training")
    print("=" * 70)

    datamodule = Folder(
        name="data_texture", 
        root=data_root, # root directory of dataset
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs, # list of abnormal class directories
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
    )

    print(f"\nInitializing PatchCore model:")
    print(f"  Backbone: {backbone}")
    print(f"  Layers: {layers}")
    print(f"  Coreset sampling ratio: {coreset_sampling_ratio}")
    print(f"  Neighbors: {num_neighbors}")

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(image_size, image_size),
        center_crop_size=(center_crop_size, center_crop_size),
    )

    model = Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        pre_processor=pre_processor,
        post_processor=True,
    )

    trainer = pl.Trainer(
        default_root_dir=out_root,
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    checkpoint_path = out_root / "patchcore_final.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    print(f"\nModel saved: {checkpoint_path}")
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)

    return model, str(checkpoint_path)


if __name__ == "__main__":
    model, checkpoint_path = train_patchcore(     # Train PatchCore
        data_dir="data",
        output_dir="outputs/patchcore",
        image_size=256,
        center_crop_size=224,
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        max_epochs=1,
        batch_size=8,
    )
