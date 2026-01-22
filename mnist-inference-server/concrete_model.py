"""
Concrete ML FHE Server Implementation for True FHE MNIST Inference.

This module provides a server wrapper for Concrete ML compiled models,
enabling true fully homomorphic encryption where the server never sees
intermediate values in plaintext.

Key features:
- Server receives encrypted inputs from clients
- All computation happens on encrypted data
- Server never decrypts intermediate values
- Only the client can decrypt the final result using their private key

Framework: Concrete ML by Zama
Scheme: TFHE (Boolean FHE)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteMNISTServer:
    """
    True FHE MNIST inference server using Concrete ML.

    This server wraps a Concrete ML compiled model and provides
    inference on encrypted data without any intermediate decryption.

    Privacy guarantees:
    - Server never sees plaintext intermediate activations
    - Server never sees the client's private key
    - Only encrypted results are returned
    - Only the client can decrypt the final result
    """

    def __init__(self, model_dir: str = "compiled_fhe_model"):
        """
        Initialize the Concrete ML FHE server.

        Args:
            model_dir: Directory containing the compiled FHE model
        """
        self.model_dir = Path(model_dir)
        self.server = None
        self.model_specs = None
        self._initialized = False

        logger.info(f"Initializing Concrete ML server from {model_dir}")

        try:
            from concrete.ml.deployment import FHEModelServer

            # Initialize FHE model server
            self.server = FHEModelServer(model_dir=str(self.model_dir))

            # Load model specifications
            specs_path = self.model_dir / "model_specs.json"
            if specs_path.exists():
                with open(specs_path, 'r') as f:
                    self.model_specs = json.load(f)

            self._initialized = True
            logger.info("Concrete ML server initialized successfully")

            # Log model info
            if self.model_specs:
                logger.info(f"Model input shape: {self.model_specs.get('input_shape', 'unknown')}")
                logger.info(f"Model output shape: {self.model_specs.get('output_shape', 'unknown')}")

        except ImportError:
            logger.error("Concrete ML not installed. Install with: pip install concrete-ml")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Concrete ML server: {e}")
            raise

    def predict(self, encrypted_input: bytes) -> bytes:
        """
        Run inference on encrypted input WITHOUT decryption.

        This is the key privacy-preserving operation - the server processes
        the encrypted data using the FHE circuit and returns an encrypted result.
        At no point does the server decrypt the intermediate values.

        Args:
            encrypted_input: Client-encrypted image data (bytes)

        Returns:
            Encrypted prediction result (bytes) - only client can decrypt

        Raises:
            RuntimeError: If server is not initialized
            ValueError: If encrypted input is invalid
        """
        if not self._initialized or self.server is None:
            raise RuntimeError("Concrete ML server not initialized")

        if not encrypted_input:
            raise ValueError("Encrypted input cannot be empty")

        try:
            logger.info("Running FHE inference on encrypted data...")

            # Run FHE inference - NO decryption happens here!
            encrypted_result = self.server.run(encrypted_input)

            logger.info(
                f"FHE inference complete. "
                f"Input size: {len(encrypted_input)} bytes, "
                f"Output size: {len(encrypted_result)} bytes"
            )

            return encrypted_result

        except Exception as e:
            logger.error(f"FHE inference failed: {e}")
            raise ValueError(f"FHE inference failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded FHE model.

        Returns:
            Dictionary with model information
        """
        info = {
            "framework": "concrete-ml",
            "scheme": "TFHE",
            "model_dir": str(self.model_dir),
            "initialized": self._initialized,
            "true_fhe": True,
            "intermediate_decryption": False,
            "privacy_level": "server_never_sees_plaintext"
        }

        if self.model_specs:
            info.update({
                "input_shape": self.model_specs.get("input_shape"),
                "output_shape": self.model_specs.get("output_shape"),
                "quantization": self.model_specs.get("quantization", {})
            })

        return info

    def verify_encrypted_input_size(self, encrypted_input: bytes) -> bool:
        """
        Verify that the encrypted input has expected size.

        Args:
            encrypted_input: Encrypted input data

        Returns:
            True if size is valid, False otherwise
        """
        # Concrete ML encrypted inputs are typically several KB
        # This is a basic sanity check
        min_size = 1000  # Minimum expected size in bytes
        max_size = 10_000_000  # Maximum expected size in bytes (10MB)

        return min_size <= len(encrypted_input) <= max_size

    def is_available(self) -> bool:
        """
        Check if the Concrete ML server is available and ready.

        Returns:
            True if server is initialized and ready
        """
        return self._initialized and self.server is not None


# Global server instance
_concrete_server: Optional[ConcreteMNISTServer] = None


def get_concrete_server(model_dir: str = "compiled_fhe_model") -> ConcreteMNISTServer:
    """
    Get or initialize the global Concrete ML server instance.

    This function implements a singleton pattern for the FHE server,
    ensuring that the model is loaded only once at startup.

    Args:
        model_dir: Directory containing the compiled FHE model

    Returns:
        Initialized ConcreteMNISTServer instance

    Raises:
        RuntimeError: If server initialization fails
    """
    global _concrete_server

    if _concrete_server is None:
        try:
            logger.info("Initializing Concrete ML server (singleton)...")
            _concrete_server = ConcreteMNISTServer(model_dir=model_dir)
            logger.info("Concrete ML server ready")
        except Exception as e:
            logger.error(f"Failed to initialize Concrete ML server: {e}")
            raise RuntimeError(f"Concrete ML server initialization failed: {e}")

    return _concrete_server


def reset_concrete_server():
    """
    Reset the global Concrete ML server instance.

    This is primarily useful for testing or when the model needs to be reloaded.
    """
    global _concrete_server
    _concrete_server = None
    logger.info("Concrete ML server instance reset")


def is_concrete_ml_available() -> bool:
    """
    Check if Concrete ML is available and properly configured.

    Returns:
        True if Concrete ML can be imported and model directory exists
    """
    try:
        from concrete.ml.deployment import FHEModelServer

        # Check if model directory exists
        model_dir = Path("compiled_fhe_model")
        if not model_dir.exists():
            logger.warning(f"Concrete ML model directory not found: {model_dir}")
            return False

        # Check for required files
        required_files = [
            "model_specs.json",
            "processed_graph.json",
            "client.zip"
        ]

        for file in required_files:
            if not (model_dir / file).exists():
                logger.warning(f"Required file not found: {model_dir / file}")
                return False

        return True

    except ImportError:
        logger.warning("Concrete ML not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking Concrete ML availability: {e}")
        return False


if __name__ == "__main__":
    """Test the Concrete ML server."""
    logger.info("Testing Concrete ML server...")

    # Check availability
    if not is_concrete_ml_available():
        logger.error("Concrete ML is not available. Please train and compile the model first:")
        logger.error("  python train_qat.py")
        exit(1)

    # Initialize server
    try:
        server = get_concrete_server()
        logger.info("Server initialized successfully")

        # Get model info
        info = server.get_model_info()
        logger.info(f"Model info: {json.dumps(info, indent=2)}")

        # Note: We cannot run actual inference without a real encrypted input
        # from a client. The server is ready to receive encrypted inputs.

        logger.info("Concrete ML server test complete!")

    except Exception as e:
        logger.error(f"Server test failed: {e}")
        exit(1)
