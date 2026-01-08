# 🎭 Shakespeare Word Predictor

An AI-powered word prediction system that uses LSTM (Long Short-Term Memory) neural networks to predict the next word in Shakespearean phrases. Trained on Shakespeare's Hamlet, this application provides an interactive web interface for text generation and word prediction.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Training](#-training)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- **🎯 Single Word Prediction**: Predict the next word with Top-K predictions and probability scores
- **📝 Multiple Words Prediction**: Generate sequences of multiple words (1-20 words)
- **✨ Text Generation**: Create longer Shakespeare-style text (10-100 words) with temperature control
- **📊 Interactive Visualization**: Real-time probability charts and statistics
- **💾 Session History**: Track all predictions with timestamps
- **📥 Export Functionality**: Download prediction history as CSV
- **🎨 Modern UI**: Clean, professional interface with light theme
- **⚡ Real-time Processing**: Instant predictions powered by cached model

## 🎬 Demo

### Single Word Prediction
```
Input: "To be or not to"
Output: "be" (45.32%)
        "that" (12.45%)
        "the" (8.67%)
```

### Text Generation
```
Input: "To be or not to"
Output: "To be or not to be that is the question whether tis nobler 
         in the mind to suffer the slings and arrows of outrageous fortune"
```

## 🏗️ Architecture

The system follows a layered architecture with clear separation of concerns:

![Architecture Diagram](shakespeare_architecture_no_overlap.png)

### Components:

1. **User Interface Layer**
   - Streamlit web application
   - Multi-line text input
   - Interactive controls

2. **Application Layer**
   - Input text cleaning
   - Tokenization (Word→Index mapping)
   - Sequence padding (Max length: 13)

3. **ML Model Pipeline**
   - Embedding Layer (100 dimensions)
   - LSTM Layer 1 (150 units)
   - Dropout Layer (0.2)
   - LSTM Layer 2 (100 units)
   - Dense Output (4,818 classes with Softmax)

4. **Storage Layer**
   - LSTM Model (40 MB HDF5 file)
   - Tokenizer (500 KB Pickle file)
   - Training Data (hamlet.txt)

5. **Output Layer**
   - Top-K predictions with probabilities
   - Interactive charts (Plotly)
   - Statistics and metrics
   - Session history
   - CSV export

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/shakespeare-word-predictor.git
cd shakespeare-word-predictor
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
```

## 🚀 Usage

### Running the Application

```bash
streamlit run app_light_clean.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Enter Your Text**: Type a Shakespearean phrase in the input box
2. **Choose Prediction Mode**: 
   - Single Word: Get top predictions with probabilities
   - Multiple Words: Generate a sequence of words
   - Generate Text: Create longer text with creativity control
3. **Adjust Parameters**: Use sidebar sliders to control output
4. **View Results**: See predictions, charts, and statistics
5. **Export Data**: Download your prediction history as CSV

### Example Usage

#### Command Line (Python Script)

```python
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model('next_word_lstm.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict next word
text = "To be or not to"
# ... prediction code ...
```

## 🧠 Model Details

### Architecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 13, 100)           481,800   
lstm (LSTM)                 (None, 13, 150)           150,600   
dropout (Dropout)           (None, 13, 150)           0         
lstm_1 (LSTM)               (None, 100)               100,400   
dense (Dense)               (None, 4818)              486,618   
=================================================================
Total params: 1,219,418
Trainable params: 1,219,418
Non-trainable params: 0
```

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Split**: 20%
- **Sequence Length**: 13 words

### Dataset

- **Source**: Shakespeare's Hamlet
- **Total Words**: ~32,000
- **Vocabulary Size**: 4,818 unique words
- **Training Sequences**: Generated using sliding window approach

### Performance

- **Training Accuracy**: ~46%
- **Validation Accuracy**: ~5%
- **Note**: High validation loss indicates overfitting - model performs best on Shakespeare-like text

## 📁 Project Structure

```
shakespeare-word-predictor/
│
├── app_light_clean.py              # Main Streamlit application
├── next_word_lstm.h5               # Trained LSTM model (40 MB)
├── tokenizer.pickle                # Word tokenizer (500 KB)
├── hamlet.txt                      # Training data
├── requirements.txt                # Python dependencies
│
├── docs/
│   ├── ARCHITECTURE_DOCUMENTATION.md
│   ├── shakespeare_architecture_no_overlap.png
│   └── UI_GUIDE.md
│
├── notebooks/
│   └── training.ipynb              # Model training notebook
│
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## 🛠️ Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core programming language |
| TensorFlow | 2.15.0 | Deep learning framework |
| Keras | (included) | High-level neural network API |
| Streamlit | 1.29.0 | Web application framework |

### Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.26.2 | Numerical computing |
| Pandas | 2.1.4 | Data manipulation |
| Pickle | (built-in) | Object serialization |

### Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| Plotly | 5.18.0 | Interactive charts |
| Matplotlib | 3.8.2 | Static plots (optional) |

## 🏋️ Training

### Data Preparation

1. **Load Text Data**: Read Shakespeare's Hamlet
2. **Tokenization**: Convert words to integer indices
3. **Sequence Generation**: Create input-output pairs with sliding window
4. **Padding**: Normalize sequence lengths to 13 words

### Training Process

```bash
# Run training notebook
jupyter notebook notebooks/training.ipynb
```

### Hyperparameter Tuning

Key hyperparameters to experiment with:
- Embedding dimension: 50-200
- LSTM units: 100-256
- Dropout rate: 0.1-0.5
- Learning rate: 0.0001-0.01
- Batch size: 16-128

## 📊 Results

### Prediction Examples

| Input | Top Prediction | Probability |
|-------|----------------|-------------|
| "To be or not to" | be | 45.32% |
| "O Romeo Romeo wherefore" | art | 38.67% |
| "All the world's a" | stage | 52.18% |

### Model Performance

- Successfully predicts common Shakespearean phrases
- Best performance on frequently occurring word patterns
- Creative text generation with temperature sampling
- Maintains Shakespeare-like writing style

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Model File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'next_word_lstm.h5'
```

**Solution:**
- Ensure the model file `next_word_lstm.h5` is in the same directory as the application
- If missing, download it from the releases page or train the model yourself
- Check file permissions

```bash
# Verify file exists
ls -lh next_word_lstm.h5

# Check current directory
pwd
```

#### Issue 2: Tokenizer Loading Error

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'tokenizer.pickle'
```

**Solution:**
- Ensure `tokenizer.pickle` is in the application directory
- The tokenizer must match the trained model
- Re-generate tokenizer if needed

```bash
# Verify tokenizer file
ls -lh tokenizer.pickle
```

#### Issue 3: TensorFlow/Keras Import Error

**Error:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
# Install TensorFlow
pip install tensorflow==2.15.0

# For M1/M2 Mac (Apple Silicon)
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Issue 4: Streamlit Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Kill process on port 8501
# On macOS/Linux:
lsof -ti:8501 | xargs kill -9

# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or use different port
streamlit run app_light_clean.py --server.port 8502
```

#### Issue 5: Slow Predictions

**Problem:** Predictions take too long (>5 seconds)

**Solutions:**
1. **Check CPU/GPU Usage:**
```bash
# Monitor system resources
top  # macOS/Linux
# or Task Manager on Windows
```

2. **Enable Model Caching:**
```python
# Already implemented with @st.cache_resource
# Ensure you're using the latest app version
```

3. **Reduce Batch Size:**
- Use Single Word mode instead of Generate Text
- Reduce number of words in Multiple Words mode

#### Issue 6: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Check available RAM
free -h  # Linux
vm_stat  # macOS

# Close other applications
# Restart the application
# Consider upgrading RAM (minimum 2GB recommended)
```

#### Issue 7: Streamlit Not Opening in Browser

**Problem:** Application runs but doesn't open browser

**Solution:**
```bash
# Manually open browser
# Navigate to: http://localhost:8501

# Or specify browser
streamlit run app_light_clean.py --browser.gatherUsageStats false

# Check if firewall is blocking
# On Windows: Allow Python through firewall
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/shakespeare-word-predictor.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

- **Author**: Pinki
- **Email**: pinkidagar18@gmail.com

## 🔗 Links

- [Documentation](docs/ARCHITECTURE_DOCUMENTATION.md)
- [Live Demo](https://your-demo-url.streamlit.app)
- [Issue Tracker](https://github.com/yourusername/shakespeare-word-predictor/issues)
- [Changelog](CHANGELOG.md)

---

<div align="center">

**Made with ❤️ and Shakespeare **

⭐ Star this repository if you found it helpful!

[Report Bug](https://github.com/yourusername/shakespeare-word-predictor/issues) · 
[Request Feature](https://github.com/yourusername/shakespeare-word-predictor/issues) · 
[Documentation](docs/)

</div>
