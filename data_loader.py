import os
import json
import pickle
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

class EnhancedDataLoader:
    def __init__(self, config):
        print("Initializing DataLoader...")
        self.config = config
        self.tokenizer = None
        self.image_ids = []
        self.captions = []
        self.image_paths = []
        self.valid_indices = []
        self.special_tokens = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3
        }

    def load_data(self, dataset_type='train'):
        print(f"\nLoading {dataset_type} data...")
        cache_file = f"{dataset_type}_cache.pkl"
        
        if os.path.exists(cache_file):
            print(f"Loading cached {dataset_type} data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self.image_ids = data['image_ids']
            self.image_paths = data['image_paths']
            self.captions = data['captions']
            self.valid_indices = data['valid_indices']
            return

        if dataset_type == 'train':
            image_dir = self.config.train_dir
            annotations_file = self.config.train_annotations
        elif dataset_type == 'val':
            image_dir = self.config.val_dir
            annotations_file = self.config.val_annotations
        else:
            raise ValueError("dataset_type must be 'train' or 'val'")

        print("Loading annotations...")
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        print("Processing annotations...")
        id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
        id_to_captions = {}

        for ann in tqdm(annotations['annotations'], desc="Processing captions"):
            image_id = ann['image_id']
            caption = ann['caption'].strip()
            if caption:  # Skip empty captions
                if image_id not in id_to_captions:
                    id_to_captions[image_id] = []
                id_to_captions[image_id].append(caption)

        print("\nValidating images...")
        valid_count = 0
        for image_id, captions in tqdm(id_to_captions.items(), desc="Checking images"):
            filename = id_to_filename.get(image_id)
            if filename:
                image_path = os.path.join(image_dir, filename)
                if os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            if (width >= self.config.min_image_size[0] and 
                                height >= self.config.min_image_size[1] and
                                width <= self.config.max_image_size[0] and
                                height <= self.config.max_image_size[1]):
                                self.image_ids.append(image_id)
                                self.image_paths.append(image_path)
                                self.captions.append(captions)
                                self.valid_indices.append(len(self.image_paths)-1)
                                valid_count += 1
                    except (IOError, OSError) as e:
                        print(f"\nSkipping corrupt image: {image_path}")
                else:
                    print(f"\nImage not found: {image_path}")

        print(f"\nLoaded {valid_count} valid {dataset_type} images (size range: {self.config.min_image_size} to {self.config.max_image_size})")

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'image_ids': self.image_ids,
                'image_paths': self.image_paths,
                'captions': self.captions,
                'valid_indices': self.valid_indices
            }, f)

    def smart_resize_with_padding(self, img):
        width, height = img.size
        target_width, target_height = self.config.target_image_size
        
        ratio = min(target_width / width, target_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        delta_w = target_width - new_size[0]
        delta_h = target_height - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        return img

    def preprocess_images(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.smart_resize_with_padding(img)
            img_array = img_to_array(img)

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array / 255.0 - mean) / std

            return img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            blank = np.zeros((*self.config.target_image_size, 3))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            return (blank - mean) / std

    def create_tokenizer(self):
        if os.path.exists(self.config.tokenizer_save_path):
            print("Loading existing tokenizer...")
            with open(self.config.tokenizer_save_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            return

        all_captions = []
        for idx in self.valid_indices:
            all_captions.extend([c for c in self.captions[idx] if c])  # Ensure no empty captions

        self.tokenizer = Tokenizer(
            num_words=self.config.vocab_size,
            oov_token="<unk>",
            filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ '
        )
        self.tokenizer.fit_on_texts(all_captions)

        # Ensure special tokens are correctly assigned
        self.tokenizer.word_index = {k: v for k, v in self.tokenizer.word_index.items()}
        for token, idx in self.special_tokens.items():
            self.tokenizer.word_index[token] = idx
            if idx in self.tokenizer.index_word:
                del self.tokenizer.index_word[idx]
        
        # Shift regular word indices to make space for special tokens
        max_special_idx = max(self.special_tokens.values())
        self.tokenizer.word_index = {
            k: v if k in self.special_tokens else v + max_special_idx + 1
            for k, v in self.tokenizer.word_index.items()
            if v < self.config.vocab_size
        }
        
        # Update index_word mapping
        self.tokenizer.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        
        # Verify special tokens
        print("Special tokens after creation:")
        print({
            token: self.tokenizer.word_index.get(token, "MISSING")
            for token in self.special_tokens
        })
        
        with open(self.config.tokenizer_save_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"Created tokenizer with {len(self.tokenizer.word_index)} words (vocab_size={self.config.vocab_size})")

    def get_sequences(self, captions):
        sequences = []
        for caption in captions:
            caption = caption.lower().strip()
            if not caption:  # Skip empty captions
                continue
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            # Filter out None values and handle OOV
            seq = [x for x in seq if x is not None]
            seq = [x if x < self.config.vocab_size + len(self.special_tokens) else self.special_tokens['<unk>']
                for x in seq]
            seq = [self.special_tokens['<start>']] + seq + [self.special_tokens['<end>']]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        return pad_sequences(
            sequences, 
            maxlen=self.config.max_caption_length, 
            padding='post', 
            truncating='post',
            value=self.special_tokens['<pad>']
        )

    def create_data_generator(self, batch_size=32, shuffle=True):
        num_samples = len(self.valid_indices)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        while True:
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_images = []
                batch_captions = []
                batch_targets = []
                
                for idx in batch_indices:
                    original_idx = self.valid_indices[idx]
                    image = self.preprocess_images(self.image_paths[original_idx])
                    batch_images.append(image)
                    
                    captions = self.captions[original_idx]
                    selected_caption = np.random.choice(captions)
                    
                    seq = self.get_sequences([selected_caption])[0]
                    input_seq = seq[:-1]  # Exclude <end>
                    target_seq = seq[1:-1]  # Exclude <start> and <end> to match model output length
                    
                    input_seq = self.pad_sequences([input_seq])[0]
                    target_seq = self.pad_sequences([target_seq])[0]
                    
                    # Ensure targets are within vocabulary size
                    target_seq = np.clip(target_seq, 0, self.config.vocab_size + len(self.special_tokens) - 1)
                    target = to_categorical(
                        target_seq,
                        num_classes=self.config.vocab_size + len(self.special_tokens)
                    )
                    
                    batch_captions.append(input_seq)
                    batch_targets.append(target)
                
                yield [np.array(batch_images), np.array(batch_captions)], np.array(batch_targets)