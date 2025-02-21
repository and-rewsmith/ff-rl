import torch
from torch import Tensor, nn
import numpy as np
from torch.nn import functional as F

INPUT_DIM = 10
BATCH_SIZE = 1
RESERVOIR_SIZE = 10
LEARNING_RATE = 0.001
LOSS_THRESHOLD = 1.5

def layer_activations_to_badness(layer_activations: torch.Tensor) -> torch.Tensor:
    badness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)

    return badness_for_layer

class FFReservoir(nn.Module):
    def __init__(self, reservoir_size: int, batch_size: int, input_dim: int, learning_rate: int, loss_threshold: float) -> None:
        super().__init__()

        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_threshold = loss_threshold

        self.weights = nn.Linear(input_dim+reservoir_size, reservoir_size, bias=True)
        self.weights.weight.data.normal_(0, 1/np.sqrt(reservoir_size))

        self.activations = torch.zeros(batch_size, reservoir_size)

        self.optimizer = torch.optim.AdamW(self.weights.parameters(), lr=learning_rate)

    def process_timestep(self, sensory_input: torch.Tensor, negative_input: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        batch_size, input_dim = sensory_input.shape
        assert (batch_size, input_dim) == (self.batch_size, self.input_dim)

        x_pos = torch.cat((sensory_input, self.activations), dim=1)
        x_pos = self.weights(x_pos)
        self.activations = F.leaky_relu(x_pos)

        x_neg = torch.cat((negative_input, self.activations), dim=1)
        x_neg = self.weights(x_neg)
        self.activations = F.leaky_relu(x_neg)

        loss = self.compute_loss(pos_act=x_pos, neg_act=x_neg)
        loss.backward()
        self.optimizer.step()

        assert self.activations.shape == (self.batch_size, self.reservoir_size)

    def compute_loss(self, pos_act: torch.Tensor, neg_act: torch.Tensor) -> torch.Tensor:
        pos_badness = layer_activations_to_badness(pos_act)
        neg_badness = layer_activations_to_badness(neg_act)

        loss: Tensor = F.softplus(torch.cat([
            (-1 * neg_badness) + self.loss_threshold,
            pos_badness - self.loss_threshold
        ])).mean()

        return loss


# sanity check for mnist training
if __name__ == "__main__":
    # sanity check with dummy tensor first
    sensory_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    negative_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    reservoir = FFReservoir(reservoir_size=RESERVOIR_SIZE, batch_size=BATCH_SIZE, input_dim=INPUT_DIM, learning_rate=LEARNING_RATE, loss_threshold=LOSS_THRESHOLD)
    reservoir.process_timestep(sensory_input, negative_input)

    print(reservoir.activations)

