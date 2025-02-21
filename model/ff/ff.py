import torch
from torch import nn
import numpy as np

"""

ff:
- init
- process_timestep (is_online)

"""

LEARNING_RATE = 0.001

class FFReservoir:
    def __init__(self, reservoir_size: int, input_dim: int, learning_rate: int) -> None:
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        self.weights = nn.Linear(input_dim, reservoir_size, bias=True)
        self.weights.weight.data.normal_(0, 1/np.sqrt(reservoir_size))

    def process_timestep(self):
        pass

# sanity check for mnist training
if __name__ == "__main__":
    pass
