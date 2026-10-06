from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class PlantPathologyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        split: str | None = None,
        transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = {"filepaths", "labels", "data set"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        if split is not None:
            self.df = self.df[self.df["data set"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.label_names = sorted(self.df["labels"].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["filepaths"]
        label_name = row["labels"]
        label_idx = self.label_to_idx[label_name]

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to read image: {image_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx, label_name, str(image_path)


def build_transforms() -> transforms.Compose:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    return weights.transforms()


def build_feature_extractor(device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
    )
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    all_embeddings = []
    all_label_idxs = []
    all_label_names: List[str] = []
    all_image_paths: List[str] = []

    for images, label_idxs, label_names, image_paths in tqdm(loader, desc="Extracting"):
        images = images.to(device, non_blocking=True)

        features = model(images)
        features = features.cpu().numpy()

        all_embeddings.append(features)
        all_label_idxs.append(label_idxs.numpy())
        all_label_names.extend(list(label_names))
        all_image_paths.extend(list(image_paths))

    embeddings = np.concatenate(all_embeddings, axis=0)
    label_idxs = np.concatenate(all_label_idxs, axis=0)

    return embeddings, label_idxs, all_label_names, all_image_paths


def save_outputs(
    out_dir: Path,
    split: str,
    embeddings: np.ndarray,
    label_idxs: np.ndarray,
    label_names: List[str],
    image_paths: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{split}_embeddings.npy", embeddings)
    np.save(out_dir / f"{split}_label_idxs.npy", label_idxs)

    meta_df = pd.DataFrame(
        {
            "image_path": image_paths,
            "label_name": label_names,
            "label_idx": label_idxs,
        }
    )
    meta_df.to_csv(out_dir / f"{split}_metadata.csv", index=False)

    print(f"Saved embeddings: {out_dir / f'{split}_embeddings.npy'}")
    print(f"Saved label indices: {out_dir / f'{split}_label_idxs.npy'}")
    print(f"Saved metadata: {out_dir / f'{split}_metadata.csv'}")
    print(f"Embedding shape: {embeddings.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract EfficientNet-B0 image embeddings.")
    parser.add_argument("--csv_path", type=str, default="data/data.csv")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/embeddings/efficientnet_b0")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = build_transforms()

    dataset = PlantPathologyDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        split=args.split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_feature_extractor(device=device)

    embeddings, label_idxs, label_names, image_paths = extract_embeddings(
        model=model,
        loader=loader,
        device=device,
    )

    save_outputs(
        out_dir=out_dir,
        split=args.split,
        embeddings=embeddings,
        label_idxs=label_idxs,
        label_names=label_names,
        image_paths=image_paths,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise