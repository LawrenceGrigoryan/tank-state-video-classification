import pickle
from pathlib import Path
from typing import NoReturn

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import vidaug.augmentors as va
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm

CLASSES = ["bridge_down", "bridge_up", "no_action", "train_in_out"]
ONNX_SAMPLE_PATH = Path(__file__).parent.joinpath('sample_onnx.pkl')
DATA_DIR = Path(__file__).parent.joinpath('../data/')


def read_clip(odir: Path,
              fname: str,
              start: int = 0,
              transposed: bool = False):
    """
    Read a video and return its frames
    """
    cpr = cv2.VideoCapture(odir.joinpath(fname).as_posix())
    has_frame = True
    frames = []

    while has_frame:
        has_frame, frame = cpr.read()
        if has_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transposed:
                frame = np.moveaxis(frame, -1, 0).copy()

            frames.append(frame)
    cpr.release()
    return np.array(frames)[start:]


def avg_video(clip: np.ndarray,
              start: int = 1,
              every_n_frame: int = 1,
              output_path: str = None) -> np.ndarray:
    """
    Averages Video into one frame
    """
    if every_n_frame > 1:
        average_frame = clip[start::every_n_frame, :, :, :].mean(axis=0)
    else:
        average_frame = clip[start:, :, :, :].mean(axis=0)
    average_frame = average_frame.astype('uint8')

    if output_path:
        cv2.imwrite(output_path, average_frame)

    return average_frame


def save_clips_npy(data: pd.DataFrame,
                   transposed: bool = False) -> NoReturn:
    """
    Save clips to np array
    """
    for _, row in tqdm(data.iterrows()):
        clip_dir = DATA_DIR.joinpath('train', row['label'])
        clip = read_clip(clip_dir, row['fname'], transposed=transposed)
        save_path = DATA_DIR.joinpath('train_np',
                                      row['label'],
                                      row['fname'].replace('.mp4', '.npy'))
        np.save(save_path, clip)


def get_class_weights(data: pd.DataFrame) -> torch.FloatTensor:
    """
    Computes class weights for given data
    """
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=CLASSES,
                                         y=data['label'])
    class_weights = torch.FloatTensor(class_weights)
    return class_weights


def predict_dataloader(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: str = 'cuda'):
    """
    Returns prediction made by given model on given dataloader
    """
    model.eval()
    model.to(device)
    preds = []
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc='Inference'):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Update statistics
            predicted = outputs.argmax(1)
            preds.extend(list(predicted.cpu().numpy()))

    return preds


def apply_video_augmentations(clip: np.ndarray,
                              p_orig: float = 0.1,
                              p_aug: float = 0.5) -> np.ndarray:
    # Choose whether to augment or keep original image:
    keep_orig = np.random.choice([True, False],
                                 p=[p_orig, 1-p_orig])
    if keep_orig:
        clip_aug = clip
    else:
        # Augmentations from albumentations
        transform = A.Compose(
            [A.RandomScale(p=p_aug, scale_limit=0.1),
             A.RandomBrightnessContrast(p=p_aug),
             A.ShiftScaleRotate(p=p_aug, shift_limit=0.0625,
                                scale_limit=0.1, rotate_limit=15),
             A.RGBShift(p=p_aug, r_shift_limit=15,
                        g_shift_limit=15, b_shift_limit=15),
             A.OneOf([
                    A.Blur(p=p_aug, blur_limit=2),
                    A.GaussNoise(p=p_aug, var_limit=5.0 / 255.0)
                ], p=p_aug),
             A.HueSaturationValue(p=p_aug),
             A.ChannelShuffle(p=p_aug),
             A.CoarseDropout(p=p_aug),
             A.Sharpen(p=p_aug),
             A.OneOf([
                A.RandomFog(p=p_aug,
                            alpha_coef=0.3,
                            fog_coef_lower=0.3,
                            fog_coef_upper=0.4),
                A.RandomSnow(p=p_aug,
                             snow_point_lower=0.2,
                             snow_point_upper=0.3,
                             brightness_coeff=1.5),
                A.RandomRain(p=p_aug,
                             brightness_coefficient=0.9,
                             drop_width=1,
                             blur_value=1),
                ], p=p_aug)],
            additional_targets={
                f'image{i}': 'image' for i in range(1, clip.shape[0])
            })
        # Apply the same augmentation to all frames of the video
        frame_dict = {f'image{i}': clip[i] for i in range(1, clip.shape[0])}
        frame_dict['image'] = clip[0]
        frames_aug = transform(**frame_dict)
        clip_aug = np.array([frames_aug[key] for key in frame_dict])

        # Augmentations from vidaug
        sometimes = lambda aug: va.Sometimes(p_aug, aug)
        seq = va.Sequential([
            sometimes(va.OneOf([va.Salt(),
                                va.Pepper()])),
            sometimes(va.RandomShear(x=0.1, y=0.1)),
            sometimes(va.HorizontalFlip()),
            # sometimes(
            #         va.OneOf([
            #             va.Downsample(ratio=np.random.choice([0.6, 0.7, 0.9])),
            #             va.Upsample(ratio=np.random.choice([1.2, 1.3, 1.5]))
            #         ])
            #     ),
        ])
        clip_aug = np.array(seq(clip_aug))

    return clip_aug
