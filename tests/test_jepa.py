from typing import Any, Dict, Optional, Tuple, cast

import pytest
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from ssl_tasks.jepa.task import JEPA as JEPABase
from ssl_tasks.tokens import TokenMask


@pytest.fixture(params=[False, True])
def loss_includes_unmasked(request):
    return request.param


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone, loss_includes_unmasked):
        class JEPA(JEPABase):
            def prepare_backbone(self, _: str) -> nn.Module:
                return backbone

            @property
            def img_size(self) -> Tuple[int, int]:
                return cast(Any, self.backbone).img_size

            def create_head(self) -> nn.Module:
                patch_size: Tuple[int, int] = cast(Any, self.backbone).patch_size
                dim: int = cast(Any, self.backbone).dim
                return nn.Linear(dim, dim)

            def create_token_mask(self, x: Tensor) -> TokenMask:
                size = x.shape[-2:]
                batch_size = x.shape[0]
                device = x.device
                return TokenMask.create(size, cast(Any, self.backbone).patch_size, batch_size, device=device)

            def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
                x = self.backbone(x, mask)
                x = self.jepa_head(x)
                return {"jepa": x}

        return JEPA(backbone, optimizer_init=optimizer_init, loss_includes_unmasked=loss_includes_unmasked)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)

    def test_predict(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.predict(task, datamodule=datamodule)
