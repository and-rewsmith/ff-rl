import torch
from torch import Tensor, nn
import numpy as np
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
INPUT_DIM = 794  # 784 image pixels + 10 for one-hot label
BATCH_SIZE = 500
RESERVOIR_SIZE = 1000
LEARNING_RATE = 0.0001
LOSS_THRESHOLD = 1.5
TIME_STEPS = 10
NUM_EPOCHS = 1000


def layer_activations_to_badness(layer_activations: torch.Tensor) -> torch.Tensor:
    badness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)
    return badness_for_layer


class FFReservoir(nn.Module):
    def __init__(
            self,
            reservoir_size: int,
            batch_size: int,
            input_dim: int,
            learning_rate: float,
            loss_threshold: float
    ) -> None:
        super().__init__()
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_threshold = loss_threshold

        self.weights = nn.Linear(input_dim+reservoir_size, reservoir_size, bias=True)
        self.weights.weight.data.normal_(0, 1/np.sqrt(reservoir_size))
        self.activations = torch.zeros(batch_size, reservoir_size, device=DEVICE)
        self.optimizer = torch.optim.AdamW(self.weights.parameters(), lr=learning_rate)

    def reset_activations(self) -> None:
        self.activations = torch.zeros(self.batch_size, self.reservoir_size, device=DEVICE)

    def reset_activations_for_batches(self, batch_indices: torch.Tensor) -> None:
        self.activations[batch_indices] = torch.zeros(
            len(batch_indices), self.reservoir_size, device=DEVICE).detach().clone()

    def process_timestep(
            self,
            sensory_input: torch.Tensor,
            negative_input: torch.Tensor,
            training: bool = True
    ) -> torch.Tensor:
        batch_size, input_dim = sensory_input.shape
        assert (batch_size, input_dim) == (self.batch_size, self.input_dim)

        if training:
            self.optimizer.zero_grad()

        x_pos = torch.cat((sensory_input, self.activations), dim=1)
        x_pos = F.leaky_relu(self.weights(x_pos))
        x_neg = torch.cat((negative_input, self.activations), dim=1)
        x_neg = F.leaky_relu(self.weights(x_neg))

        loss = self.compute_loss(pos_act=x_pos, neg_act=x_neg)
        if training:
            loss.backward()
            self.optimizer.step()

        self.activations = x_pos.clone().detach()

        return layer_activations_to_badness(x_pos)

    def compute_loss(self, pos_act: torch.Tensor, neg_act: torch.Tensor) -> torch.Tensor:
        pos_badness = layer_activations_to_badness(pos_act)
        neg_badness = layer_activations_to_badness(neg_act)

        loss: Tensor = F.softplus(torch.cat([
            (-1 * neg_badness) + self.loss_threshold,
            pos_badness - self.loss_threshold
        ])).mean()

        return loss

    def validate_batch(self, images: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        assert images.shape == (batch_size, 784)

        lowest_energies = torch.full((batch_size,), float('inf'), device=DEVICE)
        best_labels = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

        for label in range(num_classes):
            labels_onehot = F.one_hot(
                torch.full((batch_size,), label, dtype=torch.long, device=DEVICE),
                num_classes=num_classes).float()
            assert labels_onehot.shape == (batch_size, num_classes)

            batch_input = torch.cat([images, labels_onehot], dim=1)
            assert batch_input.shape == (batch_size, self.input_dim)

            self.reset_activations()
            cumulative_energy = torch.zeros(batch_size, device=DEVICE)

            for _ in range(TIME_STEPS):
                energy = self.process_timestep(batch_input, batch_input, training=False)
                cumulative_energy += energy

            better_predictions = cumulative_energy < lowest_energies
            lowest_energies[better_predictions] = cumulative_energy[better_predictions]
            best_labels[better_predictions] = label

        return best_labels, lowest_energies


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    reservoir = FFReservoir(
        reservoir_size=RESERVOIR_SIZE,
        batch_size=BATCH_SIZE,
        input_dim=INPUT_DIM,
        learning_rate=LEARNING_RATE,
        loss_threshold=LOSS_THRESHOLD
    ).to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            images_flat = images.view(BATCH_SIZE, -1)
            labels_onehot = F.one_hot(labels, num_classes=10).float()
            positive_input = torch.cat([images_flat, labels_onehot], dim=1)

            wrong_labels = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
            wrong_labels = (wrong_labels + 1 + labels) % 10
            wrong_labels_onehot = F.one_hot(wrong_labels, num_classes=10).float()
            negative_input = torch.cat([images_flat, wrong_labels_onehot], dim=1)

            reservoir.reset_activations()

            for _ in range(TIME_STEPS):
                reservoir.process_timestep(positive_input, negative_input)

            if batch_idx % 100 == 0:
                correct = 0
                total = 0

                for val_images, val_labels in val_loader:
                    val_images = val_images.to(DEVICE)
                    val_labels = val_labels.to(DEVICE)
                    val_images_flat = val_images.view(-1, 784)
                    predictions, energies = reservoir.validate_batch(val_images_flat, num_classes=10)
                    correct += (predictions == val_labels).sum().item()
                    total += val_labels.size(0)

                accuracy = correct / total
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Validation Accuracy = {accuracy:.4f}")
