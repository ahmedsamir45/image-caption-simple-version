import tensorflow as tf
import numpy as np
from memory_check import check_memory

def test_environment():
    print("=== Testing AI Environment ===")
    
    # Test TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available devices: {tf.config.list_physical_devices()}")
    
    # Test memory
    check_memory()
    
    # Test basic operations
    print("Testing basic TensorFlow operations...")
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    c = tf.matmul(a, b)
    print(f"Matrix multiplication test: {c.numpy()}")
    
    # Test model creation
    print("Testing model creation...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    print("Model created successfully!")
    
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()
