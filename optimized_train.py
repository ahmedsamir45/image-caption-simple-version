import os
import gc
import tensorflow as tf
from memory_check import check_memory, recommend_batch_size

# Memory optimization
def optimize_memory():
    # Force CPU usage for weak devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Enable mixed precision for memory efficiency
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except:
        print("Mixed precision not available, using default precision")
    
    # Set memory limit for TensorFlow
    try:
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
        print("GPU memory growth enabled")
    except:
        print("No GPU detected or GPU memory optimization not needed")

def cleanup_memory():
    gc.collect()
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    print("=== Starting Optimized Training ===")
    
    # Check memory and optimize
    check_memory()
    optimize_memory()
    
    # Import and run training
    from config import Config
    from data_loader import EnhancedDataLoader
    from model import EnhancedImageCaptioningModel
    from train import ImageCaptioningTrainer
    
    # Adjust batch size based on available memory
    config = Config()
    recommended_batch = recommend_batch_size()
    if recommended_batch < config.batch_size:
        print(f"Adjusting batch size from {config.batch_size} to {recommended_batch}")
        config.batch_size = recommended_batch
    
    # Run training with memory cleanup
    try:
        trainer = ImageCaptioningTrainer()
        trainer.config = config
        trainer.run()
    except Exception as e:
        print(f"Training error: {e}")
        cleanup_memory()
        raise
    finally:
        cleanup_memory()
