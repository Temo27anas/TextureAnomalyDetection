# Simple benchmark script to train and test PatchCore.
# Usage: python benchmark_patchcore.py

import sys
from pathlib import Path

# import training and testing functions
from src.train_patchcore import train_patchcore
from src.test_patchcore import test_patchcore

def main():
    """Run PatchCore training and testing pipeline"""
    
    print("\n" + "=" * 70)
    print("TEXTURE ANOMALY DETECTION - PATCHCORE PIPELINE")
    print("=" * 70 + "\n")
    
    # configuration
    data_dir = "data"
    output_dir = "outputs/patchcore"
    results_dir = "outputs/patchcore_results"
    
    # step 1: training
    print("STEP 1: Training PatchCore Model")
    print("-" * 70)
    
    try:
        _, checkpoint_path = train_patchcore(
            data_dir=data_dir,
            output_dir=output_dir,
            image_size=256,
            center_crop_size=224,
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            max_epochs=1,
            batch_size=8,
        )
        print("\n[OK] Training completed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # step 2: Testing
    print("\n\nSTEP 2: Testing and Evaluating Model")
    print("-" * 70)
    
    try:
        results = test_patchcore(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            output_dir=results_dir,
        )
        print("\n[OK] Testing completed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Results saved to: {results_dir}")
    print("\nGenerated files:")
    print(f"  - {results_dir}/evaluation_results.json")
    print("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
