"""
Model handling module for Indian Sign Language Interpreter.
Handles model loading and prediction functionality.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Tuple, Optional
import logging
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignLanguageModel:
    """Handles the sign language recognition model operations."""
    
    def __init__(self, model_path: str = 'model.h5'):
        """
        Initialize the model handler.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (64, 64)  # Standard input size for the model
        self.classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
    def load(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return False
            
            self.model = load_model(self.model_path)
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for prediction.
        
        Args:
            frame (np.ndarray): Input frame from video feed
            
        Returns:
            np.ndarray: Preprocessed frame ready for model prediction
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to expected input size
            resized = cv2.resize(gray, self.input_shape)
            
            # Normalize pixel values
            normalized = resized / 255.0
            
            # Reshape for model input
            processed = normalized.reshape(1, *self.input_shape, 1)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            return None
    
    def predict_sign(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Predict the sign from a video frame.
        
        Args:
            frame (np.ndarray): Input frame from video feed
            
        Returns:
            tuple: (predicted_sign, confidence_score) or (None, None) if prediction fails
        """
        try:
            if self.model is None:
                if not self.load():
                    return None, None
            
            # Preprocess the frame
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return None, None
            
            # Make prediction
            predictions = self.model.predict(processed_frame)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return self.classes[predicted_class], confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not loaded"
        
        # Redirect model summary to string
        from io import StringIO
        import sys
        
        output = StringIO()
        sys.stdout = output
        self.model.summary()
        sys.stdout = sys.__stdout__
        
        return output.getvalue()