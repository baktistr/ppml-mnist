"""
Homomorphic Encryption Utilities using TenSEAL (Microsoft SEAL CKKS Scheme)

This module provides client-side encryption/decryption for privacy-preserving ML inference.
Uses CKKS (Cheon-Kim-Kim-Song) scheme for approximate arithmetic on encrypted real numbers.

TenSEAL Documentation: https://github.com/OpenMined/TenSEAL
Microsoft SEAL: https://github.com/microsoft/SEAL
"""

import tenseal
import numpy as np
import base64
import json
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomomorphicEncryption:
    """
    Homomorphic Encryption wrapper using TenSEAL CKKS scheme.

    CKKS Scheme:
    - Supports approximate arithmetic on real numbers
    - Allows addition, multiplication, rotation, and matrix operations on ciphertexts
    - Suitable for neural network inference on encrypted data
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = [60, 40, 40, 60],
        global_scale: float = 2**40,
        security_level: int = 128
    ):
        """
        Initialize TenSEAL context with CKKS scheme.

        Args:
            poly_modulus_degree: Degree of polynomial modulus (8192 for good performance/security)
            coeff_mod_bit_sizes: Bit sizes for coefficient modulus primes
            global_scale: Scale for encoding floating-point numbers
            security_level: Security level (128-bit recommended)
        """
        logger.info(f"Initializing TenSEAL CKKS context (poly_modulus={poly_modulus_degree})")

        # Create encryption context
        self.context = tenseal.context(
            scheme=tenseal.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )

        # Set security level
        self.context.global_scale = global_scale
        self.context.generate_galois_keys()

        # For client-side: generate keys
        self._secret_key = None
        self._public_key = None

        # Store parameters for client initialization
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.security_level = security_level

        logger.info("TenSEAL CKKS context initialized")

    def generate_keys(self) -> Tuple[str, str]:
        """
        Generate public/private key pair.

        Returns:
            Tuple of (public_key_base64, secret_key_base64)
        """
        logger.info("Generating TenSEAL key pair...")

        # In TenSEAL, keys are part of the context
        # We serialize the public key for the client
        public_key = self.context.serialize(save_secret_key=False)

        # For secret key, we need to serialize with it
        secret_context = self.context.serialize(save_secret_key=True)

        self._public_key_b64 = base64.b64encode(public_key).decode('utf-8')
        self._secret_key_b64 = base64.b64encode(secret_context).decode('utf-8')

        logger.info("Key pair generated")
        return self._public_key_b64, self._secret_key_b64

    def get_public_key(self) -> str:
        """Get the public key for client use."""
        if self._public_key_b64 is None:
            self.generate_keys()
        return self._public_key_b64

    def load_context_from_public_key(self, public_key_b64: str):
        """
        Load TenSEAL context from public key (for server-side).

        Args:
            public_key_b64: Base64-encoded public key
        """
        logger.info("Loading TenSEAL context from public key")
        public_key_bytes = base64.b64decode(public_key_b64)
        self.context = tenseal.context_from(public_key_bytes)

    def load_context_from_secret_key(self, secret_key_b64: str):
        """
        Load TenSEAL context from secret key (for client-side decryption).

        Args:
            secret_key_b64: Base64-encoded secret key
        """
        logger.info("Loading TenSEAL context from secret key")
        secret_key_bytes = base64.b64decode(secret_key_b64)
        self.context = tenseal.context_from(secret_key_bytes)

    def encrypt_image(self, image_data: np.ndarray) -> str:
        """
        Encrypt a 28x28 MNIST image using CKKS.

        Args:
            image_data: numpy array of shape (28, 28) or flattened (784,)
                       Values should be in range [0, 255]

        Returns:
            Base64-encoded encrypted tensor
        """
        logger.info(f"Encrypting image (shape: {image_data.shape})")

        # Flatten and normalize to [0, 1]
        if image_data.ndim == 2:
            image_flat = image_data.flatten().astype(np.float64)
        else:
            image_flat = image_data.astype(np.float64)

        normalized = image_flat / 255.0

        # Create plain tensor
        plain_tensor = tenseal.PlainTensor(normalized, [1, 784])

        # Encrypt
        encrypted_tensor = tenseal.CKKSVector(self.context, normalized)

        # Serialize to base64
        encrypted_bytes = encrypted_tensor.serialize()
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode('utf-8')

        logger.info(f"Image encrypted: {len(encrypted_b64)} chars")
        return encrypted_b64

    def decrypt_result(self, encrypted_result_b64: str) -> dict:
        """
        Decrypt an encrypted prediction result.

        Args:
            encrypted_result_b64: Base64-encoded encrypted result

        Returns:
            Dictionary with decrypted prediction data
        """
        logger.info("Decrypting result")

        # Deserialize encrypted result
        encrypted_bytes = base64.b64decode(encrypted_result_b64)

        # Decrypt using CKKSVector
        encrypted_vector = tenseal.CKKSVector.load(self.context, encrypted_bytes)
        decrypted = encrypted_vector.decrypt()

        # Convert to list
        result = decrypted.tolist()

        logger.info(f"Result decrypted: {len(result)} values")
        return result

    def encrypt_prediction_result(self, prediction: int, confidence: float, probabilities: list) -> str:
        """
        Encrypt a prediction result for secure transmission to client.

        Args:
            prediction: Predicted digit (0-9)
            confidence: Confidence score
            probabilities: List of 10 probabilities

        Returns:
            Base64-encoded encrypted result
        """
        logger.info(f"Encrypting prediction result: digit={prediction}")

        # Combine result into single array
        result_array = np.array([float(prediction), confidence] + probabilities, dtype=np.float64)

        # Encrypt
        encrypted_vector = tenseal.CKKSVector(self.context, result_array)

        # Serialize
        encrypted_bytes = encrypted_vector.serialize()
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode('utf-8')

        logger.info(f"Prediction encrypted: {len(encrypted_b64)} chars")
        return encrypted_b64

    def decrypt_prediction_result(self, encrypted_result_b64: str) -> dict:
        """
        Decrypt a prediction result from the server.

        Args:
            encrypted_result_b64: Base64-encoded encrypted result

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Deserialize
        encrypted_bytes = base64.b64decode(encrypted_result_b64)
        encrypted_vector = tenseal.CKKSVector.load(self.context, encrypted_bytes)

        # Decrypt
        decrypted = encrypted_vector.decrypt()
        result_array = np.array(decrypted)

        # Parse result
        prediction = int(round(result_array[0]))
        confidence = float(result_array[1])
        probabilities = result_array[2:12].tolist()

        logger.info(f"Decrypted prediction: digit={prediction}, confidence={confidence:.4f}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }


# Global HE instance for server
_he_instance: Optional[HomomorphicEncryption] = None


def get_he_instance() -> HomomorphicEncryption:
    """Get or create the global HE instance."""
    global _he_instance
    if _he_instance is None:
        logger.info("Creating global HE instance")
        _he_instance = HomomorphicEncryption()
        _he_instance.generate_keys()
    return _he_instance


def initialize_he_on_startup():
    """Initialize HE system on server startup."""
    logger.info("Initializing Homomorphic Encryption system")
    he = get_he_instance()
    public_key = he.get_public_key()
    logger.info(f"HE system ready. Public key length: {len(public_key)} chars")
    return he
