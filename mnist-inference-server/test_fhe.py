"""
Test script for FHE MNIST CNN inference
This tests the complete FHE pipeline with actual MNIST data
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

logging.basicConfig(level=logging.INFO)
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


def test_fhe_with_sample_data():
    """Test FHE model with sample MNIST data"""
    print("=" * 60)
    print("Testing FHE MNIST CNN Implementation")
    print("=" * 60)

    # Initialize TenSEAL context
    print("\n1. Initializing TenSEAL CKKS context...")
    he = HomomorphicEncryption(
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
        global_scale=2**40
    )
    he.generate_keys()
    secret_key = he._secret_key  # Get secret key for decryption (from he object)
    print(f"   Context initialized with poly_modulus_degree=32768")

    # Create FHE model
    print("\n2. Loading FHE model with encrypted weights...")
    try:
        fhe_model = create_fhe_model(he.context, 'mnist_cnn.pth')
        print("   FHE model loaded successfully")
    except Exception as e:
        print(f"   ERROR loading FHE model: {e}")
        return False

    # Load test data
    print("\n3. Loading MNIST test data...")
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

    # Load regular model for comparison
    print("\n4. Loading regular CNN model for comparison...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regular_model = MNISTCNN().to(device)
    regular_model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    regular_model.eval()
    print(f"   Regular model loaded on {device}")

    # Test on a few samples
    print("\n5. Running inference on test samples...")
    print("-" * 60)

    num_test_samples = 5
    correct = 0

    for i in range(num_test_samples):
        # Get sample
        image, label = test_dataset[i]

        # Denormalize for FHE model (FHE expects 0-1 range)
        image_denorm = image * 0.3081 + 0.1307
        image_np = image_denorm.numpy().flatten() * 255.0

        print(f"\nSample {i+1}:")
        print(f"  True label: {label}")

        # Regular model prediction
        with torch.no_grad():
            output = regular_model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            regular_pred = int(torch.argmax(probs, 1).item())
            regular_conf = float(probs[0][regular_pred].item())
        print(f"  Regular CNN: prediction={regular_pred}, confidence={regular_conf:.4f}")

        # FHE model prediction
        try:
            # Encrypt input
            encrypted_input = fhe_model.preprocess_input(image_np)

            # Run FHE inference
            result = fhe_model.predict(encrypted_input, secret_key)

            fhe_pred = result['prediction']
            fhe_conf = result['confidence']

            print(f"  FHE CNN:     prediction={fhe_pred}, confidence={fhe_conf:.4f}")

            if fhe_pred == label:
                correct += 1
                print(f"  Result: CORRECT")
            else:
                print(f"  Result: INCORRECT")

        except Exception as e:
            print(f"  FHE CNN: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print("-" * 60)
    print(f"\nFHE Model Accuracy on {num_test_samples} samples: {correct}/{num_test_samples} ({100*correct/num_test_samples:.1f}%)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_fhe_with_sample_data()
    if success:
        print("\nFHE test completed successfully!")
    else:
        print("\nFHE test failed!")
