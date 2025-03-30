# Indian Sign Language Interpreter

A real-time Indian Sign Language interpreter using Machine Learning, built with Python, OpenCV, TensorFlow/Keras, and Flask.

## Features

- Real-time sign language interpretation using webcam feed
- Modern web interface for interaction
- Recognition history tracking
- Achieves 85% accuracy in real-time recognition
- Downloadable recognition history

## Tech Stack

- **Backend**: Python, Flask
- **ML Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3 (Tailwind CSS), JavaScript
- **Data Processing**: NumPy, Pandas

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python src/server.py
   ```
2. Open your web browser and navigate to `http://localhost:8000`
3. Click "Start" to begin sign language interpretation
4. Use "Pause" or "Stop" to control the interpreter
5. View recognition history in the sidebar
6. Download history as CSV if needed

## Model Training

The model was trained on a dataset from Kaggle containing Indian Sign Language images. To retrain the model:

1. Download the dataset
2. Run the training script:
   ```bash
   python src/train_model.py
   ```

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│    ├── __init__.py
│    ├── train_model.py
│    ├── model.py
│    ├── live_interpreter.py
│    └── server.py
├── templates/
│    └── index.html
└── static/
     ├── css/
     │    └── style.css
     └── js/
          └── script.js
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
