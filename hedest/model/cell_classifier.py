from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.models import resnet18

from hedest.model.base_cell_classifier import BaseCellClassifier


class ConvNet(nn.Module):
    """
    A simple Convolutional Neural Network.

    Attributes:
        layers: Sequential layer containing convolutional, batch normalization, ReLU, and pooling layers.
    """

    def __init__(self):
        """
        Initializes the ConvNet model.
        """

        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ConvNet layers.
        """

        return torch.flatten(self.layers(x), 1)


class CellClassifier(BaseCellClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embed_size: Optional[int] = None,
        image_size: Optional[tuple[int, int, int]] = None,
        hidden_dims: list = [512, 256],
        norm: bool = False,
        dropout: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Image classifier using a pretrained model from timm or a custom ConvNet.

        Args:
            model_name (str): Name of the pretrained model (e.g., "default", "convnet"
                              or "resnet18"...).
            num_classes (int): Number of output classes.
            embed_size (Optional[int]): Size of the input embeddings (if model_name is "default").
            image_size (Optional[tuple[int, int, int]]): Size of the input images (if model_name is not "default").
            hidden_dims (list): List of hidden dimensions for the fully connected layers.
            norm (bool): Whether to add a LayerNorm layer.
            dropout (float): Dropout rate.
            device (torch.device): Device to run the model on.
        """

        super().__init__(num_classes, device)
        self.model_name = model_name
        self.hidden_dims = hidden_dims
        self.norm = norm
        self.dropout = dropout
        self.embed_size = embed_size
        self.image_size = image_size

        C, H, W = self.image_size if self.image_size is not None else (None, None, None)
        self.size_edge = H

        if self.model_name == "default":
            if self.embed_size is None:
                raise ValueError("embed_size must be provided for embedding-based model")

            self.backbone = nn.Sequential()
            input_dim = self.embed_size
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.backbone.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
                if self.norm:
                    self.backbone.add_module(f"layernorm_{i}", nn.LayerNorm(hidden_dim))
                self.backbone.add_module(f"relu_{i}", nn.ReLU())
                if self.dropout > 0.0:
                    self.backbone.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
                input_dim = hidden_dim

            self.backbone.add_module("final", nn.Linear(input_dim, num_classes))

        elif self.model_name == "convnet":
            if self.size_edge is None:
                raise ValueError("image_size must be provided for image-based model")

            conv = ConvNet()

            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.size_edge, self.size_edge)
                feat = conv(dummy)
                flat_dim = feat.view(1, -1).shape[1]

            self.backbone = nn.Sequential(
                conv,
                nn.Flatten(),
                nn.Linear(flat_dim, flat_dim // 4),
                nn.ReLU(),
                nn.Linear(flat_dim // 4, self.num_classes),
            )

        elif self.model_name == "resnet18":

            resnet = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone.add_module("flatten", nn.Flatten())

            input_dim = 512
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.backbone.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
                self.backbone.add_module(f"relu_{i}", nn.ReLU())
                input_dim = hidden_dim

            self.backbone.add_module("final", nn.Linear(input_dim, num_classes))

        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)

        return F.softmax(features, dim=1)
