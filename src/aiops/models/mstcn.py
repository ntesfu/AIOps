from __future__ import annotations


class TorchNotAvailableError(RuntimeError):
    pass


def _load_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise TorchNotAvailableError(
            "PyTorch is required for MS-TCN++. Install the 'ml' extra and use a CUDA-enabled environment for training."
        ) from exc
    return torch, nn, functional


def build_ms_tcn_plus_plus(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 128,
    num_stages: int = 4,
    num_layers: int = 8,
    dropout: float = 0.3,
):
    torch, nn, functional = _load_torch()

    class DilatedResidualLayer(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.conv_dilated = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            self.conv_pointwise = nn.Conv1d(channels, channels, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = functional.relu(self.conv_dilated(x))
            out = self.conv_pointwise(out)
            return x + self.dropout(out)

    class SingleStageTCN(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.layers = nn.ModuleList(
                [DilatedResidualLayer(hidden_dim, dilation=2**layer_index) for layer_index in range(num_layers)]
            )
            self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

        def forward(self, x):
            out = self.input_projection(x)
            for layer in self.layers:
                out = layer(out)
            return self.classifier(out)

    class MultiStageTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_stage = SingleStageTCN(input_dim)
            self.refinement_stages = nn.ModuleList(
                [SingleStageTCN(num_classes) for _ in range(max(0, num_stages - 1))]
            )

        def forward(self, x):
            outputs = []
            out = self.first_stage(x)
            outputs.append(out)
            for stage in self.refinement_stages:
                out = stage(functional.softmax(out, dim=1))
                outputs.append(out)
            return torch.stack(outputs)

    return MultiStageTCN()

