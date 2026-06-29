import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class TverskyLoss(_Loss):
    """
    Tversky loss para penalizar FN más que FP.

    alpha: peso de FP  (bajo  → menos penalización de FP)
    beta:  peso de FN  (alto  → más  penalización de FN)
    alpha + beta = 1.0

    Con alpha=0.3, beta=0.7:
      → FN penalizados 2.3x más que FP
      → Empuja el modelo hacia recall alto

    Cuando alpha=beta=0.5 se comporta igual que Dice estándar.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 apply_nonlin=None, batch_dice: bool = False,
                 do_bg: bool = False, smooth: float = 1e-5, ddp: bool = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, pred, target):
        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(pred)

        # pred:   [B, C, ...]   probabilities tras softmax/sigmoid
        # target: [B, C, ...]   one-hot float

        axes = tuple(range(2, pred.ndim))  # dimensiones espaciales

        # Si batch_dice, agrupa todo el batch en una sola fracción
        if self.batch_dice:
            axes = (0,) + axes

        tp = (pred * target).sum(dim=axes)
        fp = (pred * (1 - target)).sum(dim=axes)
        fn = ((1 - pred) * target).sum(dim=axes)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Ignorar background (clase 0) si do_bg=False
        if not self.do_bg:
            tversky = tversky[:, 1:] if tversky.ndim > 1 else tversky

        return (1 - tversky).mean()


class nnUNetTrainerTversky_a03_100epochs(nnUNetTrainer):
    """
    Trainer nnUNetv2 con Tversky loss y 100 epochs.

    alpha=0.3, beta=0.7
      → FN penalizados 2.3x más que FP
      → Objetivo: maximizar recall para usar como detector
        de primera etapa en ensemble con modelo preciso.

    Uso:
        nnUNetv2_train DATASET_ID 3d_fullres_smallpatch 0 \\
            -tr nnUNetTrainerTversky_a03_100epochs

    Resultados en:
        nnUNet_results/DatasetXXX_*/nnUNetTrainerTversky_a03_100epochs/...
    → nombre único, NO sobreescribe ningún trainer existente.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    def _build_loss(self):
        loss = TverskyLoss(
            alpha=0.3,
            beta=0.7,
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions
                         else softmax_helper_dim1,
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=self.label_manager.has_regions,
            smooth=1e-5,
            ddp=self.is_ddp,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            )
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
