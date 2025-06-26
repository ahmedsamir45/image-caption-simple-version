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
