import numpy as np
import torch.nn as nn
import torch.distributions as td


class DenseModel(nn.Module):
    def __init__(
        self,
        output_shape: tuple,
        input_size: int,
        info: dict,
    ) -> None:
        """
        :param output_shape: tuple of ints, indicating the shape of the output
        :param input_size: int, length of the input vector
        :param info: dict, containing the following keys: n_layers, node_size, activation function, output_distribution etc.
        """
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._n_layers = info["n_layers"]
        self._node_size = info["node_size"]
        self.activation_fn = info["activation"]
        self.dist = info["dist"]
        self.model = self._build_model()

    def _build_model(self):
        model = [
            nn.Linear(self._input_size, self._node_size),
            self.activation_fn(),
        ]
        for _ in range(self._n_layers - 1):
            model += [
                nn.Linear(self._node_size, self._node_size),
                self.activation_fn(),
            ]
        model += [nn.Linear(self._node_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, input):
        out_dist = self.model(input)

        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(out_dist, 1), len(self._output_shape))
        if self.dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=out_dist), len(self._output_shape))
        if self.dist is None:
            return out_dist

        return NotImplementedError(self.dist)
         
    
    