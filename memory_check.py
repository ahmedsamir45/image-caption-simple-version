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
