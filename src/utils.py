import pickle
from pathlib import Path
from typing import NoReturn

import cv2
import torch
import numpy as np

onnx_sample_path = 'sample_onnx.pkl'


def read_clip(odir: Path,
              fname: str,
              start: int = 0,
              transposed: bool = True):
    """Прочесть ролик в массив."""

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


def save2onnx(model: torch.nn.Module,
              model_path: str) -> NoReturn:
    # Read sample
    with open(onnx_sample_path, 'rb') as fp:
        sample = pickle.load(fp)

    model.eval()
    model.to('cpu')
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            model_path,
            export_params=True,
            opset_version=11,
            input_names=['avg_video'],
            output_names=['output']
        )
