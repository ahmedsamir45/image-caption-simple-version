import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

class EnhancedTrainingVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def set_model(self, model):
        self.model = model  # Required by Keras

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        for key in self.history.keys():
            if key in logs:
                self.history[key].append(logs[key])

        with open(self.config.train_history_path, 'wb') as f:
            pickle.dump(self.history, f)

        self.plot_progress(epoch)

    def plot_progress(self, epoch):
        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        if self.history['val_accuracy']:
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.config.train_images_dir, f"training_progress_epoch_{epoch+1}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path