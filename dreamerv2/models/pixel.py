from typing import Tuple
import numpy as np
import torch
import torch.distirbutions as td
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int], embedding_size: int, info):
        """
        :param input_shape: tuple of ints, indicating the shape of the input
        :param embedding_size: int, length of the embedding vector
        """
        super(ObsEncoder, self).__init__()
        self.input_shape = input_shape
        activation_fn = info["activation"]
        self.depth = info["depth"]
        self.kernel = info["kernel"]
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], self.depth, self.kernel),
            activation_fn(),
            nn.Conv2d(self.depth, 2 * self.depth, self.kernel),
            activation_fn(),
            nn.Conv2d(2 * self.depth, 4 * self.depth, self.kernel),
            activation_fn()
        )

        if embedding_size == self.embed_size:
            self.fc1 = nn.Identity()
        else:
            self.fc1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, x):
        
        
