"""
Debug script to understand the FHE CNN intermediate values
"""

import torch
import torch.nn as nn
import numpy as np
import tenseal
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fhe_cnn import create_fhe_model
from encryption import HomomorphicEncryption
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MNISTCNN(nn.Module):
    """CNN Model for MNIST classification (plaintext version for comparison)"""

    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout2(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def debug_fhe_forward():
    """Debug FHE forward pass to understand intermediate values"""
    print("=" * 60)
    print("Debugging FHE CNN Forward Pass")
    print("=" * 60)

    # Initialize TenSEAL context
    print("\n1. Initializing TenSEAL CKKS context...")
    he = HomomorphicEncryption(
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
        global_scale=2**40
    )
    he.generate_keys()
    secret_key = he._secret_key

    # Create FHE model
    print("\n2. Creating FHE model...")
    fhe_model = create_fhe_model(he.context, 'mnist_cnn.pth')

    # Load regular model
    print("\n3. Loading regular CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regular_model = MNISTCNN().to(device)
    regular_model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    regular_model.eval()

    # Load test data
    print("\n4. Loading MNIST test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Test on one sample
    image, label = test_dataset[0]
    print(f"\n5. Testing sample with label: {label}")

    # Denormalize for FHE model
    image_denorm = image * 0.3081 + 0.1307
    image_np = image_denorm.numpy().flatten() * 255.0

    # Regular model forward pass with intermediate values
    print("\n6. Regular CNN forward pass (with intermediate values):")
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)

        # Layer 1
        x1 = regular_model.conv1(x)
        print(f"  After Conv1: min={x1.min().item():.4f}, max={x1.max().item():.4f}, mean={x1.mean().item():.4f}")

        x1_act = torch.relu(x1)
        print(f"  After ReLU1: min={x1_act.min().item():.4f}, max={x1_act.max().item():.4f}, mean={x1_act.mean().item():.4f}")

        x1_pool = regular_model.pool(x1_act)
        print(f"  After Pool1: min={x1_pool.min().item():.4f}, max={x1_pool.max().item():.4f}, mean={x1_pool.mean().item():.4f}")

        # Layer 2
        x2 = regular_model.conv2(x1_pool)
        print(f"  After Conv2: min={x2.min().item():.4f}, max={x2.max().item():.4f}, mean={x2.mean().item():.4f}")

        x2_act = torch.relu(x2)
        print(f"  After ReLU2: min={x2_act.min().item():.4f}, max={x2_act.max().item():.4f}, mean={x2_act.mean().item():.4f}")

        x2_pool = regular_model.pool(x2_act)
        print(f"  After Pool2: min={x2_pool.min().item():.4f}, max={x2_pool.max().item():.4f}, mean={x2_pool.mean().item():.4f}")

        # Flatten
        x_flat = x2_pool.view(-1, 64 * 7 * 7)
        print(f"  After Flatten: min={x_flat.min().item():.4f}, max={x_flat.max().item():.4f}, mean={x_flat.mean().item():.4f}")

        # FC1
        x_fc1 = regular_model.fc1(x_flat)
        print(f"  After FC1: min={x_fc1.min().item():.4f}, max={x_fc1.max().item():.4f}, mean={x_fc1.mean().item():.4f}")

        x_fc1_act = torch.relu(x_fc1)
        print(f"  After ReLU FC1: min={x_fc1_act.min().item():.4f}, max={x_fc1_act.max().item():.4f}, mean={x_fc1_act.mean().item():.4f}")

        # FC2
        logits = regular_model.fc2(x_fc1_act)
        print(f"  Final Logits: {logits[0].cpu().numpy()}")

    print("\n" + "=" * 60)
    print("The issue is that Square Activation causes values to explode!")
    print("ReLU: max(x, 0) keeps values bounded")
    print("Square: x² causes exponential growth for large values")
    print("=" * 60)


if __name__ == "__main__":
    debug_fhe_forward()
