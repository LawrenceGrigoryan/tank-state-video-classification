"""
Contains PL model classes
"""
from typing import NoReturn, Tuple, Dict, Union

import torch
import pytorch_lightning as pl
from torchmetrics import F1Score
from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    resnet18
)
from torchvision.models.video import s3d, S3D_Weights

NUM_CLASSES = 4


def construct_mn3_model(freeze_pretrained: bool = True) -> torch.nn.Module:
    """
    Create MobileNetV3 model with classification head
    """
    model = mobilenet_v3_small(weights='IMAGENET1K_V1')

    if freeze_pretrained:
        for params in model.parameters():
            params.requires_grad = False

    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


def construct_mn2_model(freeze_pretrained: bool = False) -> torch.nn.Module:
    """
    Create MobileNetV2 model with classification head
    """
    model = mobilenet_v2(weights='IMAGENET1K_V1')

    if freeze_pretrained:
        for params in model.parameters():
            params.requires_grad = False

    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


def construct_resnet18_model(
        freeze_pretrained: bool = False
        ) -> torch.nn.Module:
    """
    Create MobileNetV3 model with classification head
    """
    model = resnet18(weights='IMAGENET1K_V1')

    if freeze_pretrained:
        for params in model.parameters():
            params.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def construct_s3d_model(freeze_pretrained: bool = False):
    """
    Create S3D mode with classification head
    """
    model = s3d(weights=S3D_Weights.KINETICS400_V1)

    if freeze_pretrained:
        for params in model.parameters():
            params.requires_grad = False

    model.classifier[-1] = torch.nn.Conv3d(
        model.classifier[-1].in_channels,
        NUM_CLASSES,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1)
    )

    return model


class MobileNetV2(pl.LightningModule):
    def __init__(self,
                 freeze_pretrained: bool = False,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 class_weights: Union[None, torch.FloatTensor] = None):
        super().__init__()
        self.model = construct_mn2_model(freeze_pretrained=freeze_pretrained)
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit':
            # Metrics
            self.f1_train = F1Score(task='multiclass', threshold=0.5,
                                    num_classes=4, average='macro')
            self.f1_val = F1Score(task='multiclass', threshold=0.5,
                                  num_classes=4, average='macro')
            # Loss fn
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def training_step(self,
                      batch: Tuple[torch.Tensor, int],
                      batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('loss_train',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_train.update(logits, labels)
        self.log('f1_train',
                 self.f1_train,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def validation_step(self,
                        batch: Tuple[torch.Tensor, int],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images)

        loss = self.loss_fn(logits, labels)
        self.log('loss_val',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_val.update(logits, labels)
        self.log('f1_val',
                 self.f1_val,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer


class MobileNetV3(pl.LightningModule):
    def __init__(self,
                 freeze_pretrained: bool = False,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 class_weights: Union[None, torch.FloatTensor] = None):
        super().__init__()
        self.model = construct_mn3_model(freeze_pretrained=freeze_pretrained)
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit':
            # Metrics
            self.f1_train = F1Score(task='multiclass', threshold=0.5,
                                    num_classes=4, average='macro')
            self.f1_val = F1Score(task='multiclass', threshold=0.5,
                                  num_classes=4, average='macro')
            # Loss fn
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def training_step(self,
                      batch: Tuple[torch.Tensor, int],
                      batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('loss_train',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_train.update(logits, labels)
        self.log('f1_train',
                 self.f1_train,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def validation_step(self,
                        batch: Tuple[torch.Tensor, int],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images)

        loss = self.loss_fn(logits, labels)
        self.log('loss_val',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_val.update(logits, labels)
        self.log('f1_val',
                 self.f1_val,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer


class ResNet18(pl.LightningModule):
    def __init__(self,
                 freeze_pretrained: bool = False,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 class_weights: Union[None, torch.FloatTensor] = None):
        super().__init__()
        self.model = construct_resnet18_model(
            freeze_pretrained=freeze_pretrained
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit':
            # Metrics
            self.f1_train = F1Score(task='multiclass', threshold=0.5,
                                    num_classes=4, average='macro')
            self.f1_val = F1Score(task='multiclass', threshold=0.5,
                                  num_classes=4, average='macro')
            # Loss fn
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def training_step(self,
                      batch: Tuple[torch.Tensor, int],
                      batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('loss_train',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_train.update(logits, labels)
        self.log('f1_train',
                 self.f1_train,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def validation_step(self,
                        batch: Tuple[torch.Tensor, int],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images)

        loss = self.loss_fn(logits, labels)
        self.log('loss_val',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_val.update(logits, labels)
        self.log('f1_val',
                 self.f1_val,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer


class S3DModel(pl.LightningModule):
    def __init__(self,
                 freeze_pretrained: bool = False,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 class_weights: Union[None, torch.FloatTensor] = None):
        super().__init__()
        self.model = construct_s3d_model(
            freeze_pretrained=freeze_pretrained
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def setup(self, stage: Union[None, str] = None) -> NoReturn:
        if stage == 'fit':
            # Metrics
            self.f1_train = F1Score(task='multiclass', threshold=0.5,
                                    num_classes=4, average='macro')
            self.f1_val = F1Score(task='multiclass', threshold=0.5,
                                  num_classes=4, average='macro')
            # Loss fn
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def training_step(self,
                      batch: Tuple[torch.Tensor, int],
                      batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('loss_train',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_train.update(logits, labels)
        self.log('f1_train',
                 self.f1_train,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def validation_step(self,
                        batch: Tuple[torch.Tensor, int],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images)

        loss = self.loss_fn(logits, labels)
        self.log('loss_val',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.f1_val.update(logits, labels)
        self.log('f1_val',
                 self.f1_val,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {'loss': loss,
                'logits': logits,
                'labels': labels}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
