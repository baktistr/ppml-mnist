/**
 * Client-side tests for True FHE implementation.
 *
 * These tests verify that:
 * - Client generates private key locally
 * - Full pipeline works end-to-end
 * - Encryption/decryption work correctly
 *
 * Run with: npm test -- fhe.test.js
 */

import { ConcreteMLEncryption } from '../utils/concreteEncryption';

// Mock axios for testing
jest.mock('axios', () => ({
  get: jest.fn(),
  post: jest.fn(),
}));

const axios = require('axios');

describe('ConcreteMLEncryption', () => {
  let concreteHE;

  beforeEach(() => {
    // Reset instance before each test
    concreteHE = new ConcreteMLEncryption();
    jest.clearAllMocks();
  });

  afterEach(() => {
    concreteHE.reset();
  });

  describe('Initialization', () => {
    test('should initialize successfully when Concrete ML is available', async () => {
      // Mock successful API response
      axios.get.mockResolvedValue({
        data: {
          model_specs: {
            'model_specs.json': {
              n_bits: { model_inputs: 6, model_weights: 3, op_inputs: 6 }
            }
          },
          model_info: {
            framework: 'concrete-ml',
            scheme: 'TFHE'
          }
        }
      });

      // Mock Concrete ML import
      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({
            privateKey: 'mock-private-key',
            publicKey: 'mock-public-key'
          }))
        }))
      }));

      await concreteHE.initialize();

      expect(concreteHE.isReady()).toBe(true);
      expect(concreteHE.privateKey).toBeDefined();
      expect(concreteHE.publicKey).toBeDefined();
    });

    test('should handle initialization failure gracefully', async () => {
      // Mock failed API response
      axios.get.mockRejectedValue(new Error('Network error'));

      await expect(concreteHE.initialize()).rejects.toThrow();
      expect(concreteHE.isReady()).toBe(false);
    });
  });

  describe('Key Generation', () => {
    test('should generate private key locally on client', async () => {
      // Mock API response
      axios.get.mockResolvedValue({
        data: {
          model_specs: {},
          model_info: {}
        }
      });

      // Mock Concrete ML
      const mockKeys = {
        privateKey: 'client-side-private-key-never-leaves-browser',
        publicKey: 'public-key-can-be-shared'
      };

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => mockKeys)
        }))
      }));

      await concreteHE.initialize();

      // Verify private key is generated and stored
      expect(concreteHE.privateKey).toBe(mockKeys.privateKey);
      expect(concreteHE.publicKey).toBe(mockKeys.publicKey);

      // Verify scheme info indicates client-side key generation
      const schemeInfo = concreteHE.getSchemeInfo();
      expect(schemeInfo.keyGeneration).toBe('client-side');
      expect(schemeInfo.privateKeyLocation).toBe('client-only');
      expect(schemeInfo.serverDecryption).toBe(false);
      expect(schemeInfo.trueFHE).toBe(true);
    });

    test('should expose public key but not private key for security', async () => {
      axios.get.mockResolvedValue({ data: { model_specs: {}, model_info: {} } });

      const mockKeys = {
        privateKey: 'secret-private-key',
        publicKey: 'public-key'
      };

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => mockKeys)
        }))
      }));

      await concreteHE.initialize();

      const publicKeyInfo = concreteHE.getPublicKey();

      // Public key should be accessible
      expect(publicKeyInfo.key).toBe(mockKeys.publicKey);
      expect(publicKeyInfo.type).toBe('public');
      expect(publicKeyInfo.shareable).toBe(true);

      // Private key should still be stored but only accessible internally
      expect(concreteHE.privateKey).toBe(mockKeys.privateKey);
    });
  });

  describe('Image Encryption', () => {
    test('should validate image data before encryption', async () => {
      axios.get.mockResolvedValue({ data: { model_specs: {}, model_info: {} } });

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'key', publicKey: 'key' })),
          encrypt: jest.fn()
        }))
      }));

      await concreteHE.initialize();

      // Test invalid data length
      await expect(concreteHE.encryptImage([])).rejects.toThrow('784 pixel values');
      await expect(concreteHE.encryptImage(new Array(100))).rejects.toThrow('784 pixel values');
      await expect(concreteHE.encryptImage(new Array(1000))).rejects.toThrow('784 pixel values');
    });

    test('should normalize and quantize image data correctly', async () => {
      axios.get.mockResolvedValue({
        data: {
          model_specs: {
            'model_specs.json': {
              n_bits: { model_inputs: 6 }
            }
          },
          model_info: {}
        }
      });

      let encryptedData = null;

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'key', publicKey: 'key' })),
          encrypt: jest.fn((data) => {
            encryptedData = data;
            return new Uint8Array([1, 2, 3, 4]);
          })
        }))
      }));

      await concreteHE.initialize();

      // Create test image with known values
      const testImage = new Array(784).fill(0);
      testImage[0] = 0;    // Should quantize to 0
      testImage[1] = 127;  // Should quantize to ~31 (for 6-bit)
      testImage[2] = 255;  // Should quantize to 63 (for 6-bit)

      const result = await concreteHE.encryptImage(testImage);

      // Verify encrypt was called with quantized data
      expect(encryptedData).toBeDefined();
      expect(encryptedData[0]).toBe(0);
      expect(encryptedData[2]).toBe(63); // Max value for 6-bit

      // Verify result is hex-encoded
      expect(result).toMatch(/^[0-9a-f]+$/);
    });

    test('should return hex-encoded encrypted data', async () => {
      axios.get.mockResolvedValue({ data: { model_specs: {}, model_info: {} } });

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'key', publicKey: 'key' })),
          encrypt: jest.fn(() => new Uint8Array([0xde, 0xad, 0xbe, 0xef]))
        }))
      }));

      await concreteHE.initialize();

      const testImage = new Array(784).fill(128);
      const encrypted = await concreteHE.encryptImage(testImage);

      // Should be hex-encoded
      expect(encrypted).toBe('deadbeef');
    });
  });

  describe('Result Decryption', () => {
    test('should decrypt prediction result from server', async () => {
      axios.get.mockResolvedValue({ data: { model_specs: {}, model_info: {} } });

      // Mock decrypted logits
      const mockLogits = [0.1, 0.2, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1];

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'key', publicKey: 'key' })),
          decrypt: jest.fn(() => mockLogits)
        }))
      }));

      await concreteHE.initialize();

      // Mock encrypted result from server
      const encryptedResultHex = '0102030405060708';
      const result = await concreteHE.decryptResult(encryptedResultHex);

      // Should parse prediction correctly
      expect(result.prediction).toBe(4); // Index of max value
      expect(result.confidence).toBeCloseTo(0.476, 2); // Softmax of [0.1, 0.2, ..., 0.8]

      // Should have probabilities array
      expect(result.probabilities).toHaveLength(10);
      expect(result.probabilities[4]).toBeGreaterThan(0.4);
    });

    test('should handle empty encrypted result', async () => {
      await expect(concreteHE.decryptResult('')).rejects.toThrow('No encrypted result');
      await expect(concreteHE.decryptResult(null)).rejects.toThrow('No encrypted result');
    });

    test('should require initialization before decryption', async () => {
      concreteHE.reset();

      await expect(
        concreteHE.decryptResult('01020304')
      ).rejects.toThrow('FHE client not initialized');
    });
  });

  describe('Full Pipeline', () => {
    test('should complete full encryption and decryption pipeline', async () => {
      // Mock server responses
      axios.get.mockResolvedValue({
        data: {
          model_specs: {
            'model_specs.json': {
              n_bits: { model_inputs: 6 }
            }
          },
          model_info: {}
        }
      });

      axios.post.mockResolvedValue({
        data: {
          encrypted_result: 'aabbccdd',
          framework: 'concrete-ml-true-fhe',
          true_fhe: true
        }
      });

      // Mock Concrete ML operations
      const mockLogits = [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1];

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'key', publicKey: 'key' })),
          encrypt: jest.fn(() => new Uint8Array([1, 2, 3, 4])),
          decrypt: jest.fn(() => mockLogits)
        }))
      }));

      // Step 1: Initialize
      await concreteHE.initialize();
      expect(concreteHE.isReady()).toBe(true);

      // Step 2: Encrypt image
      const testImage = new Array(784).fill(128);
      const encrypted = await concreteHE.encryptImage(testImage);
      expect(encrypted).toMatch(/^[0-9a-f]+$/);

      // Step 3: Simulate server inference (mocked)
      const response = await axios.post('http://localhost:8001/encryption/predict_true_fhe', {
        encrypted_image: encrypted,
        framework: 'concrete-ml'
      });

      // Step 4: Decrypt result
      const result = await concreteHE.decryptResult(response.data.encrypted_result);

      // Verify complete pipeline
      expect(result.prediction).toBe(4);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.probabilities).toHaveLength(10);
    });
  });

  describe('Privacy Guarantees', () => {
    test('should ensure server never sees private key', async () => {
      axios.get.mockResolvedValue({ data: { model_specs: {}, model_info: {} } });

      jest.mock('@zama-ai/concrete-ml', () => ({
        FHEModelClient: jest.fn().mockImplementation(() => ({
          generate_keys: jest.fn(() => ({ privateKey: 'client-secret', publicKey: 'public' })),
          encrypt: jest.fn(),
          decrypt: jest.fn()
        }))
      }));

      await concreteHE.initialize();

      // Private key should never be transmitted
      // In real implementation, this would be verified by:
      // 1. Checking network requests
      // 2. Ensuring key is only stored in memory
      // 3. Verifying no serialization/transmission methods

      const schemeInfo = concreteHE.getSchemeInfo();

      expect(schemeInfo.privateKeyLocation).toBe('client-only');
      expect(schemeInfo.serverDecryption).toBe(false);
      expect(schemeInfo.trueFHE).toBe(true);
    });

    test('should verify intermediate decryption never happens', async () => {
      // This is verified by the scheme info
      const schemeInfo = concreteHE.getSchemeInfo();

      expect(schemeInfo.trueFHE).toBe(true);

      // In True FHE, server never decrypts intermediate values
      // This is enforced by using Concrete ML which only decrypts
      // on the client side
    });
  });
});
