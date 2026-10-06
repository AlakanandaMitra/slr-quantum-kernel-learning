from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class PlantPathologyDataset(Dataset):
    def __init__(self, csv_path, image_dir, split=None, transform=None):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.df = pd.read_csv(self.csv_path)

        if split is not None:
            self.df = self.df[self.df["data set"] == split].reset_index(drop=True)

        self.label_names = sorted(self.df["labels"].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["filepaths"]
        label_name = row["labels"]
        label_idx = self.label_to_idx[label_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx, label_name, str(image_path)