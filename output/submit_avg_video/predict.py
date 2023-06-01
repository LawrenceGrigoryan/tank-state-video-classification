import warnings
import pathlib
import pickle

import torchvision
import numpy as np
import onnxruntime as ort

from .avg_video import avg_video

warnings.filterwarnings("ignore")

DEVICE = 'cpu'
ID2LABEL_PATH = pathlib.Path(__file__).parent.joinpath('id2label.pkl')
ONNX_MODEL_PATH = pathlib.Path(__file__).parent.joinpath('mobilenet_v3_v1.onnx')


def load_onnx_model():
    ort_sess = ort.InferenceSession(ONNX_MODEL_PATH)
    return ort_sess


with open(ID2LABEL_PATH, 'rb') as fp:
    id2label = pickle.load(fp)

model = load_onnx_model()


def predict(clip: np.ndarray) -> str:
    """
    Вычислить класс для этого клипа.
    Эта функция должна возвращать *имя* класса.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((180, 180)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    clip_avg = avg_video(clip)
    clip_avg = transforms(clip_avg)
    clip_avg = clip_avg.unsqueeze(0)
    pred_int = model.run(
        None,
        {'avg_video': clip_avg.numpy()}
    )[0].argmax(1)[0]
    pred_label = id2label[pred_int]

    return pred_label
