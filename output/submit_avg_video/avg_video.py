import cv2
import numpy as np


def avg_video(clip: np.ndarray,
              mode: str = 'average',
              max_frames: int = 1000,
              output_path: str = None) -> np.ndarray:
    num_frames = 0
    average_frame = None
    max_frame = None
    min_frame = None
    median_frame = None
    median_median_frame = None

    for frame in clip:
        if frame is None:
            break
        if mode == "median":
            if median_frame is None:
                median_frame = np.array([frame])
            else:
                median_frame = np.append(
                    median_frame, np.array([frame]), axis=0
                )
                if num_frames % 50 == 0:
                    if median_median_frame is None:
                        median_median_frame = np.array(
                            [np.median(median_frame, axis=0)]
                        )
                        median_frame = None
                    else:
                        median_median_frame = np.append(
                            median_median_frame,
                            np.array([np.median(median_frame, axis=0)]),
                            axis=0,
                        )
                        median_frame = None
        else:
            if average_frame is None:
                average_frame = frame.astype(float)
                max_frame = frame
                min_frame = frame
            else:
                average_frame += frame.astype(float)
                max_frame = np.maximum(frame, max_frame)
                min_frame = np.minimum(frame, min_frame)
        num_frames += 1
        if num_frames >= max_frames:
            break

    if mode == "average":
        average_frame /= num_frames
        average_frame = average_frame.astype("uint8")
        output_frame = average_frame
    elif mode == "max":
        output_frame = max_frame
    elif mode == "min":
        output_frame = min_frame
    else:
        if median_median_frame is None:
            median_median_frame = np.array([np.median(median_frame, axis=0)])
        elif num_frames % 50 > 25:
            median_median_frame = np.append(
                median_median_frame,
                np.array([np.median(median_frame, axis=0)]),
                axis=0,
            )
        median_median_frame = np.median(median_median_frame, axis=0)
        output_frame = median_median_frame

    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    if output_path:
        cv2.imwrite(output_path, output_frame)

    return output_frame
