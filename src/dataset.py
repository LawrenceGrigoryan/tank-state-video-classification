import pickle
from typing import Callable, Tuple, NoReturn, Union, List
from pathlib import Path

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torchvision
import vidaug
import vidaug.augmentors as va
from torch.utils.data import Dataset, DataLoader

from .utils import read_clip
from .avg_video import avg_video

DATA_DIR = Path('../data/')
ID2LABEL_PATH = Path(__file__).parent.joinpath('id2label.pkl')

with open(ID2LABEL_PATH, 'rb') as fp:
    id2label = pickle.load(fp)
    label2id = {v: k for k, v in id2label.items()}


class AvgVideoDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 augmentations: List[vidaug.augmentors] = None,
                 transforms: Callable = None):
        self.data = data
        self.augmentations = augmentations
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get file name and label/folder
        fname = self.data.iloc[idx]['fname']
        label_str = self.data.iloc[idx]['label']

        # Read clip
        clip_path = DATA_DIR.joinpath('train', label_str, fname)
        clip = read_clip(clip_path)

        # Augmentations
        if self.augmentations:
            augmentation = np.random.choice(self.augmentations)
            if self.augmentation:
                clip = np.array(augmentation(clip))

        # Get averaged video and its integer label
        clip_avg = avg_video(clip)
        label = label2id[label_str]

        if self.transforms:
            clip_avg = self.transforms(clip_avg)

        return clip_avg, label


class AvgVideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 batch_size: int = 16,
                 resize: Tuple[int, int] = (180, 180),
                 num_workers: int = 0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers

    def setup(self, stage: Union[None, str]) -> NoReturn:
        if stage == 'fit' or stage is None:
            augmentations = [
                None,
                va.RandomCrop(size=(240, 260)),
                va.GaussianBlur(sigma=1),
                va.Add(value=20),
                va.Add(value=-20),
                va.HorizontalFlip()
            ]
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.resize),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
            self.train_dataset = AvgVideoDataset(self.train_data,
                                                 augmentations=augmentations,
                                                 transforms=transforms)
            self.val_dataset = AvgVideoDataset(self.val_data,
                                               augmentations=None,
                                               transforms=transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
