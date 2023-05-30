import cv2
import numpy as np


def avg_video(clip_path: str,
              mode: str = 'average',
              max_frames: int = 1000,
              output_path: str = None) -> np.ndarray:

    cap = cv2.VideoCapture(clip_path)

    if max_frames:
        max_frames = max_frames
    else:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames = 0
    average_frame = None
    max_frame = None
    min_frame = None
    median_frame = None
    median_median_frame = None

    while cap.isOpened():
        _, frame = cap.read()
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
        # if num_frames == 1 or num_frames % 1000 == 0 or num_frames >= max_frames:
        #     print(f"Processed frame {num_frames}/{max_frames}")
        if num_frames >= max_frames:
            break

    if not output_path:
        output_image = clip_path + "." + mode + ".jpg"
    else:
        output_image = output_path

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
        # cv2.imwrite(output_image, np.median(median_median_frame, axis=0))
        median_median_frame = np.median(median_median_frame, axis=0)
        output_frame = median_median_frame

    cap.release()
    return output_frame
