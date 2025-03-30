"""
Live interpreter module for Indian Sign Language recognition.
Handles video capture and real-time sign language interpretation.
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, List
from .model import SignLanguageModel
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveInterpreter:
    """Handles real-time video capture and sign language interpretation."""
    
    def __init__(self):
        """Initialize the live interpreter."""
        self.model = SignLanguageModel()
        self.capture = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.history: List[Dict] = []
        self.capture_thread = None
        self.process_thread = None
        
    def initialize(self) -> bool:
        """
        Initialize the video capture and model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load the model
            if not self.model.load():
                logger.error("Failed to load model")
                return False
            
            # Initialize video capture
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                logger.error("Failed to open video capture")
                return False
            
            logger.info("Interpreter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
    
    def start(self) -> bool:
        """
        Start the interpreter.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Interpreter is already running")
            return False
        
        try:
            if not self.initialize():
                return False
            
            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_frames)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            logger.info("Interpreter started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting interpreter: {str(e)}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the interpreter and release resources."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        
        if self.process_thread:
            self.process_thread.join()
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info("Interpreter stopped")
    
    def _capture_frames(self):
        """Continuously capture frames from the video feed."""
        while self.is_running:
            try:
                if self.capture is None:
                    break
                
                ret, frame = self.capture.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Update frame queue, dropping old frame if necessary
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except (queue.Empty, queue.Full):
                        pass
                        
            except Exception as e:
                logger.error(f"Error in capture thread: {str(e)}")
                break
    
    def _process_frames(self):
        """Process captured frames and perform sign language interpretation."""
        while self.is_running:
            try:
                # Get the latest frame
                frame = self.frame_queue.get(timeout=1.0)
                
                # Perform prediction
                sign, confidence = self.model.predict_sign(frame)
                
                if sign and confidence:
                    # Add to history
                    prediction = {
                        'timestamp': datetime.now().isoformat(),
                        'sign': sign,
                        'confidence': confidence
                    }
                    self.history.append(prediction)
                    
                    # Prepare frame with overlay
                    frame = self._overlay_prediction(frame, sign, confidence)
                
                # Update result queue
                try:
                    self.result_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(frame)
                    except (queue.Empty, queue.Full):
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in process thread: {str(e)}")
                break
    
    def _overlay_prediction(self, frame: np.ndarray, sign: str, confidence: float) -> np.ndarray:
        """
        Overlay prediction results on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            sign (str): Predicted sign
            confidence (float): Confidence score
            
        Returns:
            np.ndarray: Frame with overlay
        """
        # Add text overlay
        text = f"Sign: {sign} ({confidence:.2%})"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest processed frame.
        
        Returns:
            Optional[np.ndarray]: Latest frame with predictions overlay
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_history(self) -> List[Dict]:
        """
        Get the prediction history.
        
        Returns:
            List[Dict]: List of predictions with timestamps
        """
        return self.history.copy()
    
    def clear_history(self):
        """Clear the prediction history."""
        self.history.clear()