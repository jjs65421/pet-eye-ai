"""Train all five diseases, evaluate, and atomically select the completed run."""
import argparse
from pathlib import Path
from train import run, DEFAULT_DATA_ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    run(epochs=args.epochs, rebuild_manifest=args.rebuild_manifest, data_root=args.data_root)
