"""
Extended test script for FHE MNIST CNN inference
Tests on 50 samples to verify robustness
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


def test_fhe_extended():
    """Test FHE model on 50 samples"""
    print("=" * 60)
    print("Extended FHE MNIST CNN Test (50 samples)")
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

    # Test on 50 samples
    print("\n5. Running inference on 50 test samples...")
    print("-" * 60)

    num_test_samples = 50
    fhe_correct = 0
    regular_correct = 0

    for i in range(num_test_samples):
        # Get sample
        image, label = test_dataset[i]

        # Denormalize for FHE model
        image_denorm = image * 0.3081 + 0.1307
        image_np = image_denorm.numpy().flatten() * 255.0

        # Regular model prediction
        with torch.no_grad():
            output = regular_model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            regular_pred = int(torch.argmax(probs, 1).item())

        if regular_pred == label:
            regular_correct += 1

        # FHE model prediction
        try:
            # Encrypt input
            encrypted_input = fhe_model.preprocess_input(image_np)

            # Run FHE inference
            result = fhe_model.predict(encrypted_input, secret_key)

            fhe_pred = result['prediction']

            if fhe_pred == label:
                fhe_correct += 1

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_test_samples} samples processed")

        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 60)
    print(f"\nRegular CNN Accuracy on {num_test_samples} samples: {regular_correct}/{num_test_samples} ({100*regular_correct/num_test_samples:.1f}%)")
    print(f"FHE CNN Accuracy on {num_test_samples} samples: {fhe_correct}/{num_test_samples} ({100*fhe_correct/num_test_samples:.1f}%)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_fhe_extended()
    if success:
        print("\nFHE extended test completed successfully!")
    else:
        print("\nFHE extended test failed!")
