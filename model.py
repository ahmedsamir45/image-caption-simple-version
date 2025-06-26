import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Embedding, Add, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        self.attention_dense = Dense(1, activation='tanh')
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # inputs: [image_features, text_features]
        # image_features: [batch_size, embedding_dim]
        # text_features: [batch_size, max_caption_length-1, embedding_dim]
        # mask: [None, text_mask] where text_mask is [batch_size, max_caption_length-1]
        
        image_features, text_features = inputs
        text_mask = mask[1] if mask is not None else None
        
        # Expand image features to match text sequence length
        image_features = tf.expand_dims(image_features, axis=1)  # [batch_size, 1, embedding_dim]
        image_features = tf.tile(image_features, [1, tf.shape(text_features)[1], 1])  # [batch_size, max_caption_length-1, embedding_dim]
        
        # Concatenate image and text features
        attention_input = tf.concat([image_features, text_features], axis=-1)  # [batch_size, max_caption_length-1, 2*embedding_dim]
        
        # Compute attention weights
        attention_weights = self.attention_dense(attention_input)  # [batch_size, max_caption_length-1, 1]
        
        # Apply mask if provided (from text_features)
        if text_mask is not None:
            mask_dtype = attention_weights.dtype
            text_mask = tf.cast(text_mask, mask_dtype)
            text_mask = tf.expand_dims(text_mask, axis=-1)  # [batch_size, max_caption_length-1, 1]
            attention_weights += (1.0 - text_mask) * tf.cast(-1e9, mask_dtype)  # Large negative value for masked positions
        
        attention_weights = tf.nn.softmax(attention_weights, axis=1)  # [batch_size, max_caption_length-1, 1]
        
        # Apply attention to text features
        context_vector = text_features * attention_weights  # [batch_size, max_caption_length-1, embedding_dim]
        return context_vector
    
    def compute_mask(self, inputs, mask=None):
        # Propagate the text mask through this layer
        if mask is not None:
            return mask[1]  # Return only the text mask
        return None

class EnhancedImageCaptioningModel:
    def __init__(self, config):
        print("Initializing ImageCaptioningModel...")
        self.config = config
        self.image_model = None
        self.text_model = None
        self.combined_model = None
        self.tokenizer = None

    def build_image_encoder(self):
        print("\nBuilding image encoder...")
        image_input = Input(shape=(*self.config.target_image_size, 3), name="image_input")
        
        print("Loading EfficientNetB0...")
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.config.target_image_size, 3),
            pooling='avg'
        )

        print("Configuring fine-tuning...")
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        print("Adding custom dense layers...")
        features = base_model(image_input)
        features = Dense(self.config.embedding_dim * 2, activation='relu')(features)
        features = Dropout(self.config.dropout_rate)(features)
        features = Dense(self.config.embedding_dim, activation='relu')(features)
        
        self.image_model = Model(image_input, features, name="image_encoder")
        print("Image encoder built successfully!")

    def build_text_decoder(self):
        print("\nBuilding text decoder...")
        text_input = Input(shape=(self.config.max_caption_length - 1,), name="text_input")
        
        print("Creating embedding layer...")
        x = Embedding(
            input_dim=self.config.vocab_size + 4,  # Account for special tokens
            output_dim=self.config.embedding_dim,
            mask_zero=True
        )(text_input)
        
        print("Adding bidirectional LSTMs...")
        x = Bidirectional(LSTM(self.config.lstm_units, return_sequences=True))(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Bidirectional(LSTM(self.config.lstm_units // 2, return_sequences=True))(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        print("Projecting to embedding dimension...")
        x = TimeDistributed(Dense(self.config.embedding_dim, activation='relu'))(x)
        
        self.text_model = Model(text_input, x, name="text_decoder")
        print("Text decoder built successfully!")

    def build_combined_model(self):
        print("\nBuilding combined model...")
        image_input = Input(shape=(*self.config.target_image_size, 3), name="image_input")
        text_input = Input(shape=(self.config.max_caption_length - 1,), name="text_input")
        
        print("Processing image features...")
        image_features = self.image_model(image_input)  # [batch_size, embedding_dim]
        
        print("Processing text features...")
        text_features = self.text_model(text_input)  # [batch_size, max_caption_length-1, embedding_dim]
        
        print("Applying attention...")
        context_vector = AttentionLayer()([image_features, text_features])  # [batch_size, max_caption_length-1, embedding_dim]
        
        print("Combining features...")
        combined = TimeDistributed(Dense(self.config.embedding_dim * 2, activation='relu'))(context_vector)
        combined = Dropout(self.config.dropout_rate)(combined)
        
        print("Creating output layer...")
        output = TimeDistributed(Dense(self.config.vocab_size + 4, activation='softmax'))(combined)  # [batch_size, max_caption_length-1, vocab_size+4]
        
        self.combined_model = Model(
            inputs=[image_input, text_input],
            outputs=output,
            name="combined_model"
        )
        
        print("Compiling model...")
        optimizer = Adam(learning_rate=self.config.learning_rate, clipnorm=1.0)
        self.combined_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Combined model built and compiled successfully!")

    def load_tokenizer(self):
        with open(self.config.tokenizer_save_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def save_model(self):
        self.combined_model.save(self.config.model_save_path)

    def load_model(self):
        if os.path.exists(self.config.model_save_path):
            print("Loading model with custom attention layer...")
            self.combined_model = load_model(
                self.config.model_save_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            return True
        return False