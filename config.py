import os

class Config:
    def __init__(self):
        # Dataset paths
        self.coco_dir = r"D:\coco2017"  # Use raw string to handle Windows paths
        self.train_dir = os.path.join(self.coco_dir, "train2017")
        self.train_annotations = os.path.join(self.coco_dir, "annotations", "captions_train2017.json")
            
        self.val_dir = os.path.join(self.coco_dir, "val2017")
        self.val_annotations = os.path.join(self.coco_dir, "annotations", "captions_val2017.json")
        
        # Model parameters - Ultra-reduced for very weak device
        self.target_image_size = (160, 160)  # Reduced from 224x224
        self.min_image_size = (59, 51)
        self.max_image_size = (224, 224)  # Reduced from 320x320
        
        # Model architecture - Ultra-reduced complexity for very weak device
        self.embedding_dim = 128  # Reduced from 256
        self.lstm_units = 256  # Reduced from 512
        self.vocab_size = 5000  # Reduced from 10000
        self.max_caption_length = 30  # Reduced from 40
        
        # Training parameters - Ultra-reduced for very weak device
        self.batch_size = 1  # Reduced from 4 - very small batch size
        self.epochs = 15  # Reduced from 30
        self.learning_rate = 0.002  # Increased for faster convergence
        self.dropout_rate = 0.2  # Reduced from 0.3
        
        # Save paths
        self.model_save_path = "enhanced_image_captioning_model.h5"
        self.tokenizer_save_path = "enhanced_tokenizer.pkl"
        self.train_history_path = "enhanced_training_history.pkl"
        self.train_images_dir = "image_train_enhanced"
        self.checkpoint_dir = "checkpoints"
        self.logs_dir = "logs"
        
        # Create directories if they don't exist
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)