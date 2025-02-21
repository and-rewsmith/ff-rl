import torch
from torch import Tensor, nn
import numpy as np
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

INPUT_DIM = 794  # 784 image pixels + 10 for one-hot label
BATCH_SIZE = 500
RESERVOIR_SIZE = 10
LEARNING_RATE = 0.001
LOSS_THRESHOLD = 1.5
TIME_STEPS = 10

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

    def reset_activations(self) -> None:
        self.activations = torch.zeros(self.batch_size, self.reservoir_size)

    def process_timestep(self, sensory_input: torch.Tensor, negative_input: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        batch_size, input_dim = sensory_input.shape
        assert (batch_size, input_dim) == (self.batch_size, self.input_dim)

        x_pos = torch.cat((sensory_input, self.activations), dim=1)
        x_pos = self.weights(x_pos)
        x_neg = torch.cat((negative_input, self.activations), dim=1)
        x_neg = self.weights(x_neg)

        loss = self.compute_loss(pos_act=x_pos, neg_act=x_neg)
        loss.backward()
        self.optimizer.step()

        self.activations = F.leaky_relu(x_pos).detach()

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
    # Set up MNIST dataset with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    reservoir = FFReservoir(
        reservoir_size=RESERVOIR_SIZE,
        batch_size=BATCH_SIZE, 
        input_dim=INPUT_DIM,
        learning_rate=LEARNING_RATE,
        loss_threshold=LOSS_THRESHOLD
    )

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Prepare inputs
        images_flat = images.view(BATCH_SIZE, -1)  # Flatten to (batch_size, 784)
        labels_onehot = F.one_hot(labels, num_classes=10).float()
        positive_input = torch.cat([images_flat, labels_onehot], dim=1)

        # Create negative samples with random wrong labels
        wrong_labels = torch.randint(0, 10, (BATCH_SIZE,))
        # Ensure wrong labels are different from correct labels
        wrong_labels = (wrong_labels + 1 + labels) % 10
        wrong_labels_onehot = F.one_hot(wrong_labels, num_classes=10).float()
        negative_input = torch.cat([images_flat, wrong_labels_onehot], dim=1)

        # Reset activations before processing batch
        reservoir.reset_activations()
        
        # Process timesteps
        for _ in range(TIME_STEPS):
            reservoir.process_timestep(positive_input, negative_input)

    print(reservoir.activations)

