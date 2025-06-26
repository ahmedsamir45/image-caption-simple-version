# Troubleshooting Guide for Weak Devices

## Quick Setup Commands

### PowerShell (Recommended)
```powershell
.\setup_ai_environment.ps1
```

### Command Prompt
```cmd
setup_ai_environment.bat
```

## Common Issues and Solutions

### 1. Out of Memory (OOM) Errors

**Symptoms:**
- `ResourceExhaustedError: OOM when allocating tensor`
- `CUDA out of memory`
- System becomes unresponsive

**Solutions:**
```python
# Use the ultra-light config
from ultra_light_config import UltraLightConfig
config = UltraLightConfig()

# Or manually reduce batch size
config.batch_size = 1  # Minimum batch size
config.target_image_size = (128, 128)  # Even smaller images
```

**Memory Monitoring:**
```bash
python memory_check.py
```

### 2. Slow Training

**Symptoms:**
- Training takes hours per epoch
- GPU utilization is low
- CPU usage is high

**Solutions:**
- Use CPU-only TensorFlow: `pip install tensorflow-cpu`
- Reduce model complexity in config
- Use smaller dataset for testing

### 3. Import Errors

**Symptoms:**
- `ModuleNotFoundError`
- `ImportError`

**Solutions:**
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Reinstall requirements
pip install -r ai_requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

### 4. CUDA/GPU Issues

**Symptoms:**
- `Could not load dynamic library 'cudart64_*.dll'`
- GPU not detected

**Solutions:**
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Or use CPU-only TensorFlow
pip uninstall tensorflow
pip install tensorflow-cpu
```

### 5. Dataset Path Issues

**Symptoms:**
- `FileNotFoundError`
- No images found

**Solutions:**
```python
# Update path in config.py
config.coco_dir = r"YOUR_COCO_PATH"  # Update this path
```

## Performance Optimization Tips

### For Very Weak Devices (< 4GB RAM)

1. **Use Ultra-Light Config:**
```python
from ultra_light_config import UltraLightConfig
config = UltraLightConfig()
```

2. **Reduce Dataset Size:**
```python
# Use only first 1000 images for testing
config.max_samples = 1000
```

3. **Use Gradient Accumulation:**
```python
# Train with batch_size=1 but accumulate gradients
config.gradient_accumulation_steps = 4
```

### For Medium Devices (4-8GB RAM)

1. **Use Standard Config:**
```python
from config import Config
config = Config()
```

2. **Monitor Memory:**
```bash
python memory_check.py
```

### For Stronger Devices (> 8GB RAM)

1. **Increase Batch Size:**
```python
config.batch_size = 8  # or higher
```

2. **Use Larger Images:**
```python
config.target_image_size = (224, 224)
```

## Environment Testing

### Test Basic Setup
```bash
python test_environment.py
```

### Test Memory
```bash
python memory_check.py
```

### Test Training
```bash
python optimized_train.py
```

## Emergency Configurations

### Ultra-Minimal Config (for very weak devices)
```python
# Add to config.py
class EmergencyConfig:
    def __init__(self):
        self.target_image_size = (96, 96)
        self.embedding_dim = 64
        self.lstm_units = 128
        self.vocab_size = 2000
        self.batch_size = 1
        self.epochs = 10
```

### CPU-Only Training
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## Monitoring Tools

### Memory Usage
```python
import psutil
memory = psutil.virtual_memory()
print(f"Available: {memory.available / (1024**3):.2f} GB")
```

### GPU Usage (if available)
```python
import tensorflow as tf
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

## Getting Help

1. **Check the logs** in the `logs` directory
2. **Monitor memory** with `python memory_check.py`
3. **Test environment** with `python test_environment.py`
4. **Use smaller dataset** for initial testing

## Recommended Workflow

1. **Setup Environment:**
   ```bash
   .\setup_ai_environment.ps1
   ```

2. **Test Environment:**
   ```bash
   python test_environment.py
   ```

3. **Check Memory:**
   ```bash
   python memory_check.py
   ```

4. **Start Training:**
   ```bash
   python optimized_train.py
   ```

5. **Monitor Progress:**
   - Check logs directory
   - Monitor memory usage
   - Watch for errors

## Emergency Commands

### Reset Environment
```bash
# Remove and recreate virtual environment
rmdir /s venv
.\setup_ai_environment.ps1
```

### Clear Memory
```python
import gc
import tensorflow as tf
gc.collect()
tf.keras.backend.clear_session()
```

### Force CPU
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
``` 