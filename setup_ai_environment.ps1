Write-Host "=== AI Image Captioning Environment Setup ===" -ForegroundColor Cyan
Write-Host "Optimized for weak devices" -ForegroundColor Yellow

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Check if virtual environment already exists
if (Test-Path "venv") {
    Write-Host "`nRemoving existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

Write-Host "`nCreating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "Installing core dependencies..." -ForegroundColor Green
pip install -r ai_requirements.txt

# Install additional dependencies for weak devices
Write-Host "`nInstalling additional optimizations for weak devices..." -ForegroundColor Green
pip install tensorflow-cpu==2.15.0  # CPU-only version for better compatibility
pip install psutil  # For memory monitoring
pip install memory-profiler  # For memory optimization

# Create optimized config for very weak devices
Write-Host "`nCreating ultra-lightweight configuration..." -ForegroundColor Green
$ultraConfig = @"
import os

class UltraLightConfig:
    def __init__(self):
        # Dataset paths
        self.coco_dir = r"D:\coco2017"
        self.train_dir = os.path.join(self.coco_dir, "train2017")
        self.train_annotations = os.path.join(self.coco_dir, "annotations", "captions_train2017.json")
        self.val_dir = os.path.join(self.coco_dir, "val2017")
        self.val_annotations = os.path.join(self.coco_dir, "annotations", "captions_val2017.json")
        
        # Ultra-lightweight parameters for very weak devices
        self.target_image_size = (160, 160)  # Even smaller images
        self.min_image_size = (59, 51)
        self.max_image_size = (224, 224)
        
        # Minimal model architecture
        self.embedding_dim = 128  # Very small embeddings
        self.lstm_units = 256  # Smaller LSTM
        self.vocab_size = 5000  # Smaller vocabulary
        self.max_caption_length = 30  # Shorter captions
        
        # Ultra-light training parameters
        self.batch_size = 2  # Very small batch size
        self.epochs = 20  # Fewer epochs
        self.learning_rate = 0.002  # Slightly higher for faster convergence
        self.dropout_rate = 0.2  # Less dropout
        
        # Save paths
        self.model_save_path = "ultra_light_model.h5"
        self.tokenizer_save_path = "ultra_light_tokenizer.pkl"
        self.train_history_path = "ultra_light_history.pkl"
        self.train_images_dir = "image_train_ultra_light"
        self.checkpoint_dir = "checkpoints_ultra"
        self.logs_dir = "logs_ultra"
        
        # Create directories
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
"@

Set-Content -Path "ultra_light_config.py" -Value $ultraConfig

# Create memory monitoring script
Write-Host "`nCreating memory monitoring script..." -ForegroundColor Green
$memoryScript = @"
import psutil
import os

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")
    return memory.available / (1024**3)

def recommend_batch_size():
    available_gb = check_memory()
    if available_gb < 2:
        return 1
    elif available_gb < 4:
        return 2
    elif available_gb < 8:
        return 4
    else:
        return 8

if __name__ == "__main__":
    print("=== Memory Analysis ===")
    recommended_batch = recommend_batch_size()
    print(f"Recommended batch size: {recommended_batch}")
"@

Set-Content -Path "memory_check.py" -Value $memoryScript

# Create training script with memory optimization
Write-Host "`nCreating optimized training script..." -ForegroundColor Green
$trainingScript = @"
import os
import gc
import tensorflow as tf
from memory_check import check_memory, recommend_batch_size

# Memory optimization
def optimize_memory():
    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU memory optimization failed: {e}")
    
    # Set memory limit for CPU
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('CPU')[0], True)
    
    # Enable mixed precision for memory efficiency
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

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
"@

Set-Content -Path "optimized_train.py" -Value $trainingScript

# Create a simple test script
Write-Host "`nCreating test script..." -ForegroundColor Green
$testScript = @"
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
"@

Set-Content -Path "test_environment.py" -Value $testScript

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Files created:" -ForegroundColor Yellow
Write-Host "- ultra_light_config.py (for very weak devices)" -ForegroundColor White
Write-Host "- memory_check.py (memory monitoring)" -ForegroundColor White
Write-Host "- optimized_train.py (memory-optimized training)" -ForegroundColor White
Write-Host "- test_environment.py (environment testing)" -ForegroundColor White

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Test environment: python test_environment.py" -ForegroundColor White
Write-Host "2. Check memory: python memory_check.py" -ForegroundColor White
Write-Host "3. Start training: python optimized_train.py" -ForegroundColor White

Write-Host "`nTo activate environment later: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Read-Host "Press Enter to continue" 