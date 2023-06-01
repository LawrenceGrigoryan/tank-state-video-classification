import torch
from torchvision.models import mobilenet_v3_small

NUM_CLASSES = 4


def construct_mn3_model(freeze_pretrained: bool = True) -> torch.nn.Module:
    model = mobilenet_v3_small(pretrained=True)

    for params in model.parameters():
        params.requires_grad = freeze_pretrained

    model.classifier[-1] = torch.nn.Linear(1024, NUM_CLASSES)

    return model
