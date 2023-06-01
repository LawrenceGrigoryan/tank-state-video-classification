import pickle
from typing import Callable, Tuple
from pathlib import Path

import torch
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path('../data/')
SRC_DIR = Path('../src/')

with open(SRC_DIR.joinpath('id2label.pkl'), 'rb') as fp:
    id2label = pickle.load(fp)
    label2id = {v: k for k, v in id2label.items()}


class AvgVideoDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 transforms: Callable = None):
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname = self.data.iloc[idx]['fname']
        label_str = self.data.iloc[idx]['label']
        clip_avg_path = DATA_DIR.joinpath('train_avg', label_str, fname)
        clip_avg_frame = torchvision.io.read_image(str(clip_avg_path))
        label = label2id[label_str]

        if self.transforms:
            clip_avg_frame = self.transforms(
                clip_avg_frame.numpy().transpose(1, 2, 0)
            )

        return clip_avg_frame, label
