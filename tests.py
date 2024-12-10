import os
import numpy as np
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Test: Directory Setup
def test_directory_setup(tmp_path):
    # Temporary directories for testing
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    os.makedirs(test_dir / "class1", exist_ok=True)
    os.makedirs(test_dir / "class2", exist_ok=True)
    
    # Add dummy files
    with open(test_dir / "class1" / "file1.txt", "w") as f:
        f.write("dummy")
    with open(test_dir / "class2" / "file2.txt", "w") as f:
        f.write("dummy")
    
    # Verify directories and files exist
    assert (test_dir / "class1" / "file1.txt").exists()
    assert (test_dir / "class2" / "file2.txt").exists()

# Test: Model Initialization
def test_model_initialization():
    # Initialize a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Validate the structure
    assert len(model.layers) == 3
    assert model.input_shape == (None, 28, 28)

# Test: Model Prediction Output
def test_prediction_output():
    # Create a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Dummy input data
    dummy_input = np.random.rand(1, 28, 28)
    prediction = model.predict(dummy_input)
    
    # Validate output shape and properties
    assert prediction.shape == (1, 10)
    assert np.isclose(np.sum(prediction), 1.0, atol=1e-3)  # Ensure probabilities sum to 1

# Test: Confusion Matrix Calculation
def test_confusion_matrix():
    # Ground truth and predicted labels
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 0, 0]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Validate the shape and specific values
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2  # True positives for class 0
    assert cm[2, 0] == 1  # Misclassified class 2 as class 0

# Test: Training Directory Contains Expected Subdirectories
def test_train_directory_structure(tmp_path):
    train_dir = tmp_path / "train"
    os.makedirs(train_dir / "class1")
    os.makedirs(train_dir / "class2")
    
    # Check subdirectories
    subdirectories = list(train_dir.iterdir())
    assert len(subdirectories) == 2
    assert (train_dir / "class1").exists()
    assert (train_dir / "class2").exists()

