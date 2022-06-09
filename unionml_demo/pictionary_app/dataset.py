from typing import List, Optional

import urllib.request
from tqdm.auto import tqdm
from pathlib import Path
import requests
import torch
import math
import numpy as np


CLASSES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def get_quickdraw_class_names():
    """Get the class names associated with the quickdraw dataset."""
    return [*sorted(x.replace(' ', '_') for x in requests.get(CLASSES_URL).text.splitlines())]


def download_quickdraw_dataset(root: str = "./data", limit: Optional[int] = None, class_names: List[str]=None):
    class_names = class_names or get_quickdraw_class_names()
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    print("Downloading Quickdraw Dataset...")
    for class_name in tqdm(class_names[:limit]):
        fpath = root / f"{class_name}.npy"
        if not fpath.exists():
            urllib.request.urlretrieve(f"{DATASET_URL}{class_name.replace('_', '%20')}.npy", fpath)


def load_quickdraw_data(root: str = "./data", max_items_per_class: int = 5000):
    x = np.empty([0, 784], dtype=np.uint8)
    y = np.empty([0], dtype=np.long)
    class_names = []
    print(f"Loading {max_items_per_class} examples for each class from the Quickdraw Dataset...")
    for idx, file in enumerate(tqdm(sorted(Path(root).glob('*.npy')))):
        data = np.load(file, mmap_mode='r')[0: max_items_per_class, :]
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, np.full(data.shape[0], idx))
        class_names.append(file.stem)
    return x, y, class_names


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_items_per_class=5000, class_limit=None):
        super().__init__()
        self.root = root
        self.max_items_per_class = max_items_per_class
        self.class_limit = class_limit
        download_quickdraw_dataset(self.root, self.class_limit)
        self.X, self.Y, self.classes = load_quickdraw_data(self.root, self.max_items_per_class)

    def __getitem__(self, idx):
        x = (self.X[idx] / 255.).astype(np.float32).reshape(1, 28, 28)
        y = self.Y[idx]
        return torch.from_numpy(x), y.item()

    def __len__(self):
        return len(self.X)

    @staticmethod
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([item[0] for item in batch]),
            'labels': torch.LongTensor([item[1] for item in batch]),
        }
    
    def split(self, pct=0.1):
        indices = torch.randperm(len(self)).tolist()
        n_val = math.floor(len(indices) * pct)
        train_ds = torch.utils.data.Subset(self, indices[:-n_val])
        val_ds = torch.utils.data.Subset(self, indices[-n_val:])
        return train_ds, val_ds
