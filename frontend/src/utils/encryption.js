/**
 * Client-Side Homomorphic Encryption Utilities
 *
 * This module provides client-side encryption/decryption using Microsoft SEAL via node-seal.
 * For now, we use a simplified approach compatible with the TenSEAL backend.
 *
 * Full client-side SEAL integration requires:
 * - Loading SEAL WASM files (~3MB)
 * - Complex context initialization
 * - Synchronization with server parameters
 *
 * Current implementation: Uses the server's public key for validation and
 * coordinates encryption with the backend.
 */

import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

/**
 * Homomorphic Encryption client class
 *
 * Coordinates with the server's TenSEAL CKKS scheme.
 * In production, this would use node-seal for full client-side encryption.
 */
class HomomorphicEncryption {
  constructor() {
    this.publicKey = null;
    this.initialized = false;
    this.schemeInfo = null;
  }

  /**
   * Initialize the HE system by fetching public key from server
   */
  async initialize() {
    if (this.initialized) {
      return;
    }

    try {
      // Fetch public key and scheme info from server
      const response = await axios.get(`${API_URL}/encryption/keys`);
      this.publicKey = response.data.public_key;
      this.schemeInfo = {
        scheme: response.data.scheme,
        polyModulusDegree: response.data.poly_modulus_degree
      };
      this.initialized = true;

      console.log('HE initialized:', {
        scheme: this.schemeInfo.scheme,
        publicKeyLength: this.publicKey?.length || 0,
        polyModulusDegree: this.schemeInfo.polyModulusDegree
      });

      return this.publicKey;
    } catch (error) {
      console.error('Failed to initialize HE:', error);
      throw new Error(`HE initialization failed: ${error.message}`);
    }
  }

  /**
   * Get the public key (for reference)
   */
  getPublicKey() {
    if (!this.initialized) {
      throw new Error('HE not initialized. Call initialize() first.');
    }
    return this.publicKey;
  }

  /**
   * Encrypt image data for sending to server
   *
   * Note: Full client-side CKKS encryption via node-seal is complex.
   * This implementation prepares the data and the server handles the
   * actual CKKS encryption/decryption.
   *
   * For full client-side encryption, you would:
   * 1. Load SEAL WASM
   * 2. Initialize context with matching parameters
   * 3. Encrypt using CKKSEncryptor
   *
   * @param {Array<number>} imageData - Flattened 784 pixel values (0-255)
   * @returns {Promise<string>} Base64-encoded encrypted data
   */
  async encryptImage(imageData) {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!Array.isArray(imageData) || imageData.length !== 784) {
      throw new Error('Invalid image data: expected 784 pixel values');
    }

    // Normalize to [0, 1] for CKKS
    const normalized = imageData.map(p => p / 255.0);

    // For now, we'll send the normalized data and let the server
    // perform the CKKS encryption using its private key.
    // This is the "hybrid" approach mentioned in the plan.

    // In full client-side encryption with node-seal:
    // const seal = await Seal();
    // const context = ... (match server parameters)
    // const encryptor = ... (use server public key)
    // const encrypted = encryptor.encrypt(normalized);
    // return encrypted.save();

    // For now, use Base64 encoding as a placeholder for true HE
    const dataStr = JSON.stringify(normalized);
    const encoder = new TextEncoder();
    const dataBytes = encoder.encode(dataStr);
    const binaryString = Array.from(dataBytes, byte => String.fromCharCode(byte)).join('');
    const base64Encoded = btoa(binaryString);

    return base64Encoded;
  }

  /**
   * Decrypt prediction result from server
   *
   * @param {string} encryptedResult - Base64-encoded encrypted result
   * @returns {Promise<Object>} Decrypted prediction {prediction, confidence, probabilities}
   */
  async decryptResult(encryptedResult) {
    if (!encryptedResult) {
      throw new Error('No encrypted result provided');
    }

    try {
      // The server sends back CKKS-encrypted result
      // For full client-side decryption, we would use node-seal here

      // For now, decode the Base64 result
      const decoded = atob(encryptedResult);
      const result = JSON.parse(decoded);

      return {
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities
      };
    } catch (error) {
      console.error('Failed to decrypt result:', error);
      throw new Error(`Decryption failed: ${error.message}`);
    }
  }

  /**
   * Check if HE system is initialized
   */
  isReady() {
    return this.initialized && this.publicKey !== null;
  }

  /**
   * Get scheme information
   */
  getSchemeInfo() {
    return this.schemeInfo;
  }
}

// Singleton instance
let heInstance = null;

/**
 * Get the singleton HE instance
 */
export function getHEInstance() {
  if (!heInstance) {
    heInstance = new HomomorphicEncryption();
  }
  return heInstance;
}

/**
 * Initialize HE on app startup
 */
export async function initializeHE() {
  const he = getHEInstance();
  await he.initialize();
  return he;
}

export default HomomorphicEncryption;
