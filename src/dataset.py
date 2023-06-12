"""
Contains classes for PL data modules
"""
import pickle
from typing import Callable, Tuple, NoReturn, Union
from pathlib import Path

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torchvision
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from utils import avg_video, apply_video_augmentations

DATA_DIR = Path(__file__).parent.joinpath('../data/')
ID2LABEL_PATH = Path(__file__).parent.joinpath('id2label.pkl')

with open(ID2LABEL_PATH, 'rb') as fp:
    id2label = pickle.load(fp)
    label2id = {v: k for k, v in id2label.items()}


class AvgVideoDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 every_n_frame: int = 1,
                 augmentations: list = None,
                 transforms: Callable = None):
        self.data = data
        self.every_n_frame = every_n_frame
        self.augmentations = augmentations
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get file name and label/folder
        fname = self.data.iloc[idx]['fname']
        label_str = self.data.iloc[idx]['label']

        # Read clip
        clip_path = DATA_DIR.joinpath('train_np', label_str, fname)
        clip = np.load(clip_path)

        # Get averaged video and its integer label (F, C, H, W) -> (C, H, W)
        clip_avg = avg_video(clip, start=1, every_n_frame=self.every_n_frame)
        label = label2id[label_str]

        # Augmentations
        if self.augmentations:
            clip_avg = self.augmentations(image=clip_avg)['image']

        # Transform image
        if self.transforms:
            clip_avg = self.transforms(clip_avg)

        return clip_avg, label


class AvgVideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 every_n_frame: int = 1,
                 batch_size: int = 16,
                 augment: bool = False,
                 num_workers: int = 0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.every_n_frame = every_n_frame

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit' or stage is None:
            # Define augmentations
            if self.augment:
                augmentations = A.Compose([
                    A.HorizontalFlip(p=0.25),
                    A.RandomBrightnessContrast(p=0.25,
                                               brightness_limit=0.1,
                                               contrast_limit=0.1),
                    A.GaussianBlur(p=0.25,
                                   blur_limit=(1, 3)),
                    A.RGBShift(p=0.25),
                    A.RandomFog(p=0.25,
                                fog_coef_lower=0.2,
                                fog_coef_upper=0.3),
                    A.Sharpen(p=0.25),
                    A.RandomSnow(p=0.25,
                                 snow_point_lower=0.2,
                                 snow_point_upper=0.3,
                                 brightness_coeff=1.5),
                    A.Affine(p=0.25, rotate=(-10, 10), shear=(-15, 15))
                ])
            else:
                augmentations = None

            # Define transforms
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop((224, 224)),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])

            # Create datasets
            self.train_dataset = AvgVideoDataset(
                self.train_data,
                every_n_frame=self.every_n_frame,
                augmentations=augmentations,
                transforms=transforms
            )
            self.val_dataset = AvgVideoDataset(
                self.val_data,
                every_n_frame=self.every_n_frame,
                augmentations=False,
                transforms=transforms
            )

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


class VideoDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 n_frames: int = 16,
                 augment: bool = False,
                 transforms: Callable = None):
        self.data = data
        self.n_frames = n_frames
        self.augment = augment
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get file name and label/folder
        fname = self.data.iloc[idx]['fname']
        label_str = self.data.iloc[idx]['label']

        # Read clip
        clip_path = DATA_DIR.joinpath('train_np', label_str, fname)
        clip = np.load(clip_path)
        label = label2id[label_str]

        # Make number of frames consistent over videos
        frame_idx = np.linspace(0, clip.shape[0],
                                self.n_frames, endpoint=False).astype(int)
        clip = clip[frame_idx, :, :, :]

        # Augmentations
        if self.augment:
            clip = apply_video_augmentations(clip)

        # Transpose to correct shape (F, H, W, C) -> (F, C, H, W)
        clip = torch.Tensor(clip.transpose(0, 3, 1, 2))
        if self.transforms:
            clip = self.transforms(clip)

        # Transpose to (C, F, H, W)
        clip = clip.transpose(1, 0)

        return clip, label


class VideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 batch_size: int = 16,
                 n_frames: int = 16,
                 augment: bool = False,
                 num_workers: int = 0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit' or stage is None:
            # Define transforms, takes (F, C, H, W)
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]
                )
            ])

            # Create datasets
            self.train_dataset = VideoDataset(
                self.train_data,
                n_frames=self.n_frames,
                augment=self.augment,
                transforms=transforms
            )
            self.val_dataset = VideoDataset(
                self.val_data,
                n_frames=self.n_frames,
                augment=False,
                transforms=transforms
            )

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
