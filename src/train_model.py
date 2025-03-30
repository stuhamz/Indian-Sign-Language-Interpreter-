"""
Training script for Indian Sign Language Interpreter model.
Handles dataset loading, preprocessing, model creation, and training.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles the training of the sign language recognition model."""
    
    def __init__(self, data_dir: str = 'dataset'):
        """
        Initialize the model trainer.
        
        Args:
            data_dir (str): Directory containing the training data
        """
        self.data_dir = data_dir
        self.input_shape = (64, 64, 1)  # Grayscale images
        self.num_classes = 26  # A-Z
        self.batch_size = 32
        self.epochs = 50
        self.validation_split = 0.2
        
    def create_model(self) -> tf.keras.Model:
        """
        Create and compile the CNN model.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            tuple: (x_train, y_train, x_val, y_val)
        """
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Data directory {self.data_dir} not found")
            
            images = []
            labels = []
            
            # Load images and labels
            for label, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                letter_dir = os.path.join(self.data_dir, letter)
                if not os.path.exists(letter_dir):
                    continue
                
                logger.info(f"Loading images for letter {letter}")
                
                for img_file in os.listdir(letter_dir):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    img_path = os.path.join(letter_dir, img_file)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize
                    img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                    
                    # Normalize
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(label)
            
            # Convert to numpy arrays
            x = np.array(images)
            y = np.array(labels)
            
            # Reshape images to include channel dimension
            x = x.reshape(-1, self.input_shape[0], self.input_shape[1], 1)
            
            # Convert labels to one-hot encoding
            y = tf.keras.utils.to_categorical(y, self.num_classes)
            
            # Split into train and validation sets
            x_train, x_val, y_train, y_val = train_test_split(
                x, y,
                test_size=self.validation_split,
                random_state=42,
                stratify=y
            )
            
            return x_train, y_train, x_val, y_val
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train(self):
        """Train the model."""
        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            x_train, y_train, x_val, y_val = self.load_and_preprocess_data()
            
            # Create data generator for augmentation
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False
            )
            
            # Create and compile model
            logger.info("Creating model...")
            model = self.create_model()
            model.summary()
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            logger.info("Starting training...")
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=self.batch_size),
                validation_data=(x_val, y_val),
                epochs=self.epochs,
                callbacks=callbacks
            )
            
            # Save model
            logger.info("Saving model...")
            model.save('model.h5')
            
            # Plot training history
            self.plot_training_history(history)
            
            # Evaluate model
            logger.info("Evaluating model...")
            test_loss, test_accuracy = model.evaluate(x_val, y_val)
            logger.info(f"Test accuracy: {test_accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history
        """
        try:
            # Create directory for plots if it doesn't exist
            os.makedirs('training_plots', exist_ok=True)
            
            # Plot accuracy
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'training_plots/training_history_{timestamp}.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise

if __name__ == '__main__':
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")