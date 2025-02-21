import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

"""

ff:
- init
- process_timestep (is_online)

"""

INPUT_DIM = 10
BATCH_SIZE = 1
RESERVOIR_SIZE = 10
LEARNING_RATE = 0.001

class FFReservoir:
    def __init__(self, reservoir_size: int, batch_size: int, input_dim: int, learning_rate: int) -> None:
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.weights = nn.Linear(input_dim+reservoir_size, reservoir_size, bias=True)
        self.weights.weight.data.normal_(0, 1/np.sqrt(reservoir_size))

        self.activations = torch.zeros(batch_size, reservoir_size)

    def process_timestep(self, sensory_input: torch.Tensor) -> None:
        batch_size, input_dim = sensory_input.shape
        assert (batch_size, input_dim) == (self.batch_size, self.input_dim)

        # forward pass
        x = torch.cat((sensory_input, self.activations), dim=1)
        x = self.weights(x)
        self.activations = F.leaky_relu(x)
        assert self.activations.shape == (self.batch_size, self.reservoir_size)

# sanity check for mnist training
if __name__ == "__main__":
    # sanity check with dummy tensor first
    sensory_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    reservoir = FFReservoir(reservoir_size=RESERVOIR_SIZE, batch_size=BATCH_SIZE, input_dim=INPUT_DIM, learning_rate=LEARNING_RATE)
    reservoir.process_timestep(sensory_input)

    print(reservoir.activations)

