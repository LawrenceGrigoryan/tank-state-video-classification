import warnings
import pathlib
import pickle
import yaml

import torchvision
import numpy as np
import onnxruntime as ort

from .avg_video import avg_video

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
    every_n_frame = config['every_n_frame']


model = load_onnx_model()


def predict(clip: np.ndarray) -> str:
    """
    Вычислить класс для этого клипа.
    Эта функция должна возвращать *имя* класса.
    """
    # Get averaged video
    clip_avg = avg_video(clip, start=1, every_n_frame=every_n_frame)
    # Apply transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    clip_avg = transforms(clip_avg)
    # Add batch dimension
    clip_avg = clip_avg.unsqueeze(0)
    # Make prediction
    output = model.run(None,
                       {'input': clip_avg.numpy()})
    pred_int = output[0].argmax(1)[0]
    pred_label = id2label[pred_int]
    return pred_label
