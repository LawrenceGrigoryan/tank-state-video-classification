from typing import NoReturn, Tuple, Dict, Union

import torch
import pytorch_lightning as pl
from torchmetrics import F1Score
from torchvision.models import mobilenet_v3_small


def construct_mn3_model(freeze_pretrained: bool = True) -> torch.nn.Module:
    model = mobilenet_v3_small(pretrained=True)

    for params in model.parameters():
        params.requires_grad = freeze_pretrained

    model.classifier[-1] = torch.nn.Linear(1024, 4)

    return model


class MobileNetV3(pl.LightningModule):
    def __init__(self,
                 freeze_pretrained: bool = True,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01):
        super().__init__()
        self.model = construct_mn3_model(freeze_pretrained)
        self.lr = lr
        self.weight_decay = weight_decay

    def setup(self, stage: Union[None, str]) -> NoReturn:
        if stage == 'fit':
            # Metrics
            self.f1_train = F1Score(task='multiclass', threshold=0.5,
                                    num_classes=4, average='macro')
            self.f1_val = F1Score(task='multiclass', threshold=0.5,
                                  num_classes=4, average='macro')
            # Loss fn
            self.loss_fn = torch.nn.CrossEntropyLoss()

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
