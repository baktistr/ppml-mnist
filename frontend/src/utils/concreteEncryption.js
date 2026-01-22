/**
 * Concrete ML Client-Side Encryption for True FHE
 *
 * This module provides client-side encryption/decryption using Concrete ML,
 * enabling true fully homomorphic encryption where:
 * - Private keys are generated and stored on the client
 * - Server never sees intermediate values in plaintext
 * - Only the client can decrypt the final prediction result
 *
 * Framework: Concrete ML by Zama (TFHE scheme)
 */

import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

class ConcreteMLEncryption {
  constructor() {
    this.fheClient = null;
    this.privateKey = null;
    this.publicKey = null;
    this.initialized = false;
    this.modelSpecs = null;
  }

  /**
   * Initialize Concrete ML client with server model specifications.
   *
   * This method:
   * 1. Fetches FHE model specs from the server
   * 2. Initializes the Concrete ML client
   * 3. Generates cryptographic keys locally on the client
   *
   * The private key NEVER leaves the client!
   */
  async initialize() {
    if (this.initialized) {
      return;
    }

    try {
      // Step 1: Fetch FHE model specification from server
      const response = await axios.get(`${API_URL}/encryption/fhe_model_spec`);
      this.modelSpecs = response.data.model_specs;

      // Step 2: Initialize Concrete ML client with model specs
      // Note: This requires the concrete-ml package
      const { FHEModelClient } = await import('@zama-ai/concrete-ml');

      this.fheClient = new FHEModelClient(this.modelSpecs);

      // Step 3: Generate keys - PRIVATE KEY STAYS ON CLIENT!
      // This is the key difference from the hybrid approach
      const keys = this.fheClient.generate_keys();
      this.privateKey = keys.privateKey;
      this.publicKey = keys.publicKey;

      this.initialized = true;

      console.log('Concrete ML initialized successfully', {
        scheme: 'TFHE',
        keyGeneration: 'client-side',
        privateKeyStored: 'client-only',
        modelSpecsLoaded: Object.keys(this.modelSpecs || {}).length > 0
      });

    } catch (error) {
      console.error('Concrete ML initialization failed:', error);

      // If Concrete ML is not available, provide helpful error
      if (error.message?.includes('Cannot find module')) {
        throw new Error(
          'Concrete ML not installed. Install with: npm install @zama-ai/concrete-ml'
        );
      }

      throw error;
    }
  }

  /**
   * Encrypt image data for True FHE inference.
   *
   * This method:
   * 1. Normalizes and quantizes the image data
   * 2. Encrypts it using the client's private key
   * 3. Returns encrypted data as hex string for transmission
   *
   * @param {Array<number>} imageData - Flattened 784 pixel values (0-255)
   * @returns {Promise<string>} Hex-encoded encrypted image
   */
  async encryptImage(imageData) {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!Array.isArray(imageData) || imageData.length !== 784) {
      throw new Error('Invalid image data: expected 784 pixel values');
    }

    try {
      // Step 1: Normalize pixel values to [0, 1]
      const normalized = imageData.map(p => p / 255.0);

      // Step 2: Quantize to integers based on model's bit width
      // Typically 6-bit quantization (values 0-63)
      const quantizationBits = this.modelSpecs?.model_specs?.['model_specs.json']?.n_bits?.model_inputs || 6;
      const maxValue = Math.pow(2, quantizationBits) - 1;
      const quantized = normalized.map(p => Math.floor(Math.min(1, Math.max(0, p)) * maxValue));

      // Step 3: Encrypt using client-side FHE
      // The encryption happens locally using the client's private key
      const encrypted = this.fheClient.encrypt(quantized);

      // Step 4: Convert to hex string for safe transmission
      // Hex encoding ensures binary-safe transmission over HTTP
      const encryptedHex = Buffer.from(encrypted).toString('hex');

      console.log('Image encrypted for True FHE', {
        inputPixels: imageData.length,
        quantizedTo: `${quantizationBits}-bit integers`,
        encryptedSize: `${encrypted.length} bytes`,
        hexEncoded: encryptedHex.length
      });

      return encryptedHex;

    } catch (error) {
      console.error('Image encryption failed:', error);
      throw new Error(`Image encryption failed: ${error.message}`);
    }
  }

  /**
   * Decrypt prediction result from the server.
   *
   * This method:
   * 1. Converts hex-encoded result back to bytes
   * 2. Decrypts using client's private key (only the client can do this!)
   * 3. Parses the prediction and converts to probabilities
   *
   * @param {string} encryptedResultHex - Hex-encoded encrypted result from server
   * @returns {Promise<Object>} Decrypted prediction {prediction, confidence, probabilities}
   */
  async decryptResult(encryptedResultHex) {
    if (!this.fheClient) {
      throw new Error('FHE client not initialized. Call initialize() first.');
    }

    if (!encryptedResultHex) {
      throw new Error('No encrypted result provided');
    }

    try {
      // Step 1: Convert hex to bytes
      const encryptedBytes = Buffer.from(encryptedResultHex, 'hex');

      // Step 2: Decrypt using client's private key
      // This only works because we have the private key!
      const decrypted = this.fheClient.decrypt(encryptedBytes);

      // Step 3: Parse prediction (convert logits to probabilities)
      // The decrypted result is typically logits for each class
      const maxLogit = Math.max(...decrypted);
      const expLogits = decrypted.map(l => Math.exp(l - maxLogit));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probabilities = expLogits.map(e => e / sumExp);

      const prediction = probabilities.indexOf(Math.max(...probabilities));
      const confidence = Math.max(...probabilities);

      console.log('Result decrypted successfully', {
        prediction,
        confidence: (confidence * 100).toFixed(1) + '%',
        decryptionPerformed: 'client-side'
      });

      return {
        prediction,
        confidence,
        probabilities
      };

    } catch (error) {
      console.error('Result decryption failed:', error);
      throw new Error(`Result decryption failed: ${error.message}`);
    }
  }

  /**
   * Check if the Concrete ML system is ready.
   *
   * @returns {boolean} True if initialized and ready
   */
  isReady() {
    return this.initialized && this.fheClient !== null;
  }

  /**
   * Get information about the encryption scheme.
   *
   * @returns {Object} Scheme information
   */
  getSchemeInfo() {
    return {
      framework: 'concrete-ml',
      scheme: 'TFHE',
      provider: 'Zama',
      keyGeneration: 'client-side',
      privateKeyLocation: 'client-only',
      serverDecryption: false,
      trueFHE: true
    };
  }

  /**
   * Get the public key (for reference/validation).
   *
   * Note: The public key can be shared with the server for verification,
   * but the private key NEVER leaves the client.
   *
   * @returns {Object} Public key information
   */
  getPublicKey() {
    if (!this.initialized) {
      throw new Error('FHE client not initialized');
    }
    return {
      key: this.publicKey,
      type: 'public',
      shareable: true
    };
  }

  /**
   * Reset the client (clear keys and state).
   *
   * This is useful for testing or when re-initialization is needed.
   */
  reset() {
    this.fheClient = null;
    this.privateKey = null;
    this.publicKey = null;
    this.initialized = false;
    this.modelSpecs = null;
    console.log('Concrete ML client reset');
  }
}

// Singleton instance
let concreteInstance = null;

/**
 * Get the singleton Concrete ML encryption instance.
 *
 * @returns {ConcreteMLEncryption} The singleton instance
 */
export function getConcreteMLEncryption() {
  if (!concreteInstance) {
    concreteInstance = new ConcreteMLEncryption();
  }
  return concreteInstance;
}

/**
 * Initialize Concrete ML on app startup.
 *
 * @returns {Promise<ConcreteMLEncryption>} The initialized instance
 */
export async function initializeConcreteML() {
  const concreteHE = getConcreteMLEncryption();
  await concreteHE.initialize();
  return concreteHE;
}

export default ConcreteMLEncryption;
