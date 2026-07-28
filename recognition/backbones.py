"""IResNet backbones for AdaFace, implemented from scratch.

Architecture
------------
This is the "improved ResNet" (IR) family introduced with ArcFace and reused by
AdaFace.  It differs from a stock ResNet in three ways that all matter for
faces:

BN-first residual blocks
    Each block begins with batch norm rather than convolution.  Faces arrive
    with wildly varying illumination, and normalising before the first
    convolution stops that variation propagating through the block.

PReLU instead of ReLU
    A learned negative slope.  ReLU discards all negative activations, which on
    a 112x112 input costs detail the network cannot recover; face recognition
    is unusually sensitive to that loss.

No global average pooling
    A stock ResNet pools its final feature map to a vector, discarding *where*
    each feature occurred.  Spatial arrangement is exactly what distinguishes
    two faces, so the IR family flattens the full 7x7x512 map and projects it
    through a single linear layer instead.

Why the norm is returned
------------------------
:meth:`AdaFaceBackbone.forward` returns both the L2-normalised embedding and
its pre-normalisation magnitude.  That magnitude is not a by-product.  AdaFace
trains with a margin scaled by feature norm, which makes the norm correlate
strongly with image quality -- sharp, well-lit, frontal faces produce large
norms and blurred or extreme-pose faces produce small ones.  Phase 7 uses it as
a quality gate, so discarding it here would mean re-deriving quality from
pixels less reliably and at greater cost.

State dictionary compatibility
------------------------------
Module names match the official AdaFace release exactly, so published
checkpoints load without key remapping.  The naming is inherited rather than
chosen; ``body.0``, ``res_layer`` and ``shortcut_layer`` are not names this
module would have picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "BlockSpec",
    "BasicBlockIR",
    "AdaFaceBackbone",
    "build_backbone",
    "SUPPORTED_ARCHITECTURES",
]

#: Backbones this module implements.
SUPPORTED_ARCHITECTURES: Tuple[str, ...] = ("ir18", "ir50")

#: Spatial size of the final feature map for a 112x112 input, after four
#: stride-2 stages: 112 -> 56 -> 28 -> 14 -> 7.
FEATURE_MAP_SIZE = 7

#: Channel width of the final stage.
FINAL_CHANNELS = 512

#: Dimensionality of the emitted embedding.
EMBEDDING_DIM = 512


@dataclass(frozen=True)
class BlockSpec:
    """Configuration for one residual stage.

    Attributes:
        in_channels: Channels entering the stage.
        out_channels: Channels leaving the stage.
        num_units: Residual blocks in the stage.  Only the first downsamples.
    """

    in_channels: int
    out_channels: int
    num_units: int


#: Stage layouts. IR-50's depth is concentrated in the third stage, at 14x14,
#: where mid-level facial structure lives.
_LAYOUTS = {
    "ir18": (
        BlockSpec(64, 64, 2),
        BlockSpec(64, 128, 2),
        BlockSpec(128, 256, 2),
        BlockSpec(256, 512, 2),
    ),
    "ir50": (
        BlockSpec(64, 64, 3),
        BlockSpec(64, 128, 4),
        BlockSpec(128, 256, 14),
        BlockSpec(256, 512, 3),
    ),
}


class BasicBlockIR(nn.Module):
    """Improved-ResNet residual block.

    Args:
        in_channels: Channels entering the block.
        out_channels: Channels leaving the block.
        stride: Spatial stride of the second convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        if in_channels == out_channels:
            # Identity shortcut. MaxPool with a 1x1 kernel is a pure stride
            # operation: it changes resolution without mixing channels or
            # adding parameters. Named to match the reference checkpoints.
            self.shortcut_layer: nn.Module = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        return self.res_layer(x) + self.shortcut_layer(x)


def _build_stage(spec: BlockSpec) -> List[nn.Module]:
    """Expand a stage specification into residual blocks.

    Only the first block of a stage changes resolution and channel count;
    the rest preserve both.

    Args:
        spec: The stage configuration.

    Returns:
        The stage's blocks in order.
    """
    blocks: List[nn.Module] = [BasicBlockIR(spec.in_channels, spec.out_channels, stride=2)]
    blocks.extend(
        BasicBlockIR(spec.out_channels, spec.out_channels, stride=1)
        for _ in range(spec.num_units - 1)
    )
    return blocks


class AdaFaceBackbone(nn.Module):
    """IResNet feature extractor emitting a normalised embedding and its norm.

    Args:
        layout: Stage specifications, from :data:`_LAYOUTS`.
        dropout: Dropout probability before the final projection.
        input_size: Expected input resolution as ``(height, width)``.

    Raises:
        ValueError: If the input size is not 112x112, which is what the
            flattened projection's dimensions assume.
    """

    def __init__(
        self,
        layout: Sequence[BlockSpec],
        dropout: float = 0.4,
        input_size: Tuple[int, int] = (112, 112),
    ) -> None:
        super().__init__()

        if input_size != (112, 112):
            raise ValueError(
                f"Only 112x112 input is supported, got {input_size}. The final "
                "projection is sized for a 7x7x512 feature map."
            )

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        blocks: List[nn.Module] = []
        for spec in layout:
            blocks.extend(_build_stage(spec))
        self.body = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(FINAL_CHANNELS),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(FINAL_CHANNELS * FEATURE_MAP_SIZE * FEATURE_MAP_SIZE, EMBEDDING_DIM),
            # affine=False: the embedding's direction carries identity, and a
            # learned per-dimension scale would let the network re-introduce a
            # magnitude the subsequent L2 normalisation only strips again.
            nn.BatchNorm1d(EMBEDDING_DIM, affine=False),
        )

        self._initialise()

    def _initialise(self) -> None:
        """Initialise weights for training from scratch.

        Overwritten wholesale when a checkpoint is loaded; present so the model
        is usable without one.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract an embedding and its pre-normalisation magnitude.

        Args:
            x: Batch of aligned faces, shape ``(B, 3, 112, 112)``, normalised
                to roughly ``[-1, 1]``.

        Returns:
            A tuple ``(embedding, norm)`` where ``embedding`` has shape
            ``(B, 512)`` and unit rows, and ``norm`` has shape ``(B, 1)`` and
            carries the quality signal.
        """
        features = self.input_layer(x)
        features = self.body(features)
        features = self.output_layer(features)

        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        # Clamp rather than add epsilon: an exactly zero feature vector would
        # otherwise produce NaNs that propagate silently into the gallery.
        embedding = features / norm.clamp(min=1e-12)
        return embedding, norm


def build_backbone(architecture: str, dropout: float = 0.4) -> AdaFaceBackbone:
    """Construct a backbone by name.

    Args:
        architecture: ``"ir18"`` or ``"ir50"``.
        dropout: Dropout probability before the final projection.

    Returns:
        The constructed backbone in evaluation mode.

    Raises:
        ValueError: If the architecture is not supported.
    """
    key = architecture.lower()
    if key not in _LAYOUTS:
        raise ValueError(
            f"Unsupported architecture {architecture!r}; "
            f"expected one of {SUPPORTED_ARCHITECTURES}"
        )
    model = AdaFaceBackbone(_LAYOUTS[key], dropout=dropout)
    model.eval()
    return model