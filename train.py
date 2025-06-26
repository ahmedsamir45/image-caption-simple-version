import time
import pickle
from tensorflow.keras.callbacks import Callback
from config import Config
from data_loader import EnhancedDataLoader
from model import EnhancedImageCaptioningModel
from visualization import EnhancedTrainingVisualizer
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

class ProgressCallback(Callback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch+1}/{self.trainer.config.epochs}")
        self.epoch_pbar = tqdm(total=self.trainer.steps_per_epoch, desc="Batch Progress")

    def on_batch_end(self, batch, logs=None):
        self.epoch_pbar.update(1)
        self.epoch_pbar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'accuracy': f"{logs['accuracy']:.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_pbar.close()
        print(f"Epoch {epoch+1} completed")
        print(f"Training Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
        if 'val_loss' in logs:
            print(f"Validation Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")

class ImageCaptioningTrainer:
    def __init__(self):
        print("Initializing ImageCaptioningTrainer...")
        self.config = Config()
        self.data_loader = EnhancedDataLoader(self.config)
        self.val_data_loader = EnhancedDataLoader(self.config)
        self.model = EnhancedImageCaptioningModel(self.config)
        self.visualizer = EnhancedTrainingVisualizer(self.config)
        
        self.steps_per_epoch = 0
        self.validation_steps = 0

    def run(self):
        try:
            self.prepare_data()
            self.build_models()
            self.train()
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise

    def prepare_data(self):
        print("\n===== Preparing Data =====")
        try:
            print("Loading training data...")
            self.data_loader.load_data(dataset_type='train')
            print("Creating tokenizer...")
            self.data_loader.create_tokenizer()
            self.model.tokenizer = self.data_loader.tokenizer
            
            print("\nLoading validation data...")
            self.val_data_loader.load_data(dataset_type='val')
            self.val_data_loader.tokenizer = self.data_loader.tokenizer
            
            print("\nCreating data generators...")
            self.train_generator = self.data_loader.create_data_generator(
                batch_size=self.config.batch_size
            )
            self.val_generator = self.val_data_loader.create_data_generator(
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            if not self.data_loader.valid_indices:
                raise ValueError("No valid training images found. Check dataset paths and image sizes.")
            if not self.val_data_loader.valid_indices:
                raise ValueError("No valid validation images found. Check dataset paths and image sizes.")
            
            self.steps_per_epoch = len(self.data_loader.valid_indices) // self.config.batch_size
            self.validation_steps = len(self.val_data_loader.valid_indices) // self.config.batch_size
            print(f"\nSteps per epoch: {self.steps_per_epoch}")
            print(f"Validation steps: {self.validation_steps}")
            
            print("\n=== Tokenizer Verification ===")
            print(f"Total tokens: {len(self.data_loader.tokenizer.word_index)}")
            print(f"Expected max: {self.config.vocab_size + 4} (vocab_size + special tokens)")
            print("Special tokens:")
            print({
                "<pad>": self.data_loader.tokenizer.word_index.get("<pad>", "MISSING"),
                "<start>": self.data_loader.tokenizer.word_index.get("<start>", "MISSING"),
                "<end>": self.data_loader.tokenizer.word_index.get("<end>", "MISSING"),
                "<unk>": self.data_loader.tokenizer.word_index.get("<unk>", "MISSING")
            })
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise

    def build_models(self):
        print("\n===== Building Models =====")
        self.model.build_image_encoder()
        self.model.build_text_decoder()
        self.model.build_combined_model()
        self.model.combined_model.summary()

    def train(self):
        print("\n===== Starting Training =====")

        checkpoint_cb = ModelCheckpoint(
            filepath=self.config.model_save_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.combined_model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config.epochs,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=[
                ProgressCallback(self),
                self.visualizer,
                checkpoint_cb,
                early_stopping
            ],
            verbose=0
        )

        print("\nTraining completed! Saving model...")
        self.model.save_model()
        print("Model saved successfully!")

if __name__ == "__main__":
    trainer = ImageCaptioningTrainer()
    trainer.run()