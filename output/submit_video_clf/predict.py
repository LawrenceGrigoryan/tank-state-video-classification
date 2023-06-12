import warnings
import pathlib
import pickle
import yaml

import torchvision
import numpy as np
import onnxruntime as ort

warnings.filterwarnings("ignore")

ID2LABEL_PATH = pathlib.Path(__file__).parent.joinpath('id2label.pkl')
ONNX_MODEL_PATH = pathlib.Path(__file__).parent.joinpath('mnv3.onnx')
CONFIG_PATH = pathlib.Path(__file__).parent.joinpath('config.yaml')


def load_onnx_model():
    ort_sess = ort.InferenceSession(ONNX_MODEL_PATH)
    return ort_sess


with open(ID2LABEL_PATH, 'rb') as fp:
    id2label = pickle.load(fp)


with open(CONFIG_PATH) as fp:
    config = yaml.safe_load(fp)
    n_frames = config['n_frames']


model = load_onnx_model()


def predict(clip: np.ndarray) -> str:
    # Make number of frames consistent over videos
    frame_idx = np.linspace(0, clip.shape[0],
                            n_frames, endpoint=False).astype(int)
    clip = clip[frame_idx, :, :, :]
    # Apply transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )
    ])
    clip = transforms(clip)
    # Add batch dimension
    clip = clip.unsqueeze(0)
    # Make prediction
    output = model.run(None,
                       {'input': clip.numpy()})
    pred_int = output[0].argmax(1)[0]
    pred_label = id2label[pred_int]

    return pred_label
