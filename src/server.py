"""
Flask server for Indian Sign Language Interpreter.
Handles web interface and video streaming.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import json
import logging
from datetime import datetime
import pandas as pd
from .live_interpreter import LiveInterpreter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.abspath('../templates'),
    static_folder=os.path.abspath('../static')
)
CORS(app)

# Initialize interpreter
interpreter = LiveInterpreter()

def generate_frames():
    """Generate frames for video streaming."""
    while True:
        frame = interpreter.get_latest_frame()
        if frame is not None:
            # Convert frame to jpg
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/start', methods=['POST'])
def start_interpreter():
    """Start the interpreter."""
    try:
        if interpreter.start():
            return jsonify({'status': 'success', 'message': 'Interpreter started'})
        return jsonify({'status': 'error', 'message': 'Failed to start interpreter'})
    except Exception as e:
        logger.error(f"Error starting interpreter: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_interpreter():
    """Stop the interpreter."""
    try:
        interpreter.stop()
        return jsonify({'status': 'success', 'message': 'Interpreter stopped'})
    except Exception as e:
        logger.error(f"Error stopping interpreter: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/history')
def get_history():
    """Get interpretation history."""
    try:
        history = interpreter.get_history()
        return jsonify({'status': 'success', 'history': history})
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/download_history')
def download_history():
    """Download interpretation history as CSV."""
    try:
        history = interpreter.get_history()
        if not history:
            return jsonify({'status': 'error', 'message': 'No history available'})
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Convert to CSV string
        csv_data = df.to_csv(index=False)
        
        # Create response
        from flask import make_response
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = 'attachment; filename=sign_language_history.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
        
    except Exception as e:
        logger.error(f"Error downloading history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear interpretation history."""
    try:
        interpreter.clear_history()
        return jsonify({'status': 'success', 'message': 'History cleared'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('../templates', exist_ok=True)
    os.makedirs('../static', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)