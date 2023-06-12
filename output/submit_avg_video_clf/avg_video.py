import cv2
import numpy as np


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
