# Arabic Font Recognition System

## Overview

This project is an arabic font recognition system implemented in Python. It consists of modules for data reading, preprocessing, feature extraction, model training, testing, and accuracy evaluation. The system is designed to recognize fonts from images and provide accuracy metrics based on the trained model.

## Modules

### 1. `readdata`

- Module for reading font images and labels from specified directories.
- Provides functions for reading data for training and testing.

### 2. `preprocessing`

- Module for preprocessing raw image data.
- Performs tasks such as resizing, normalization, and noise reduction.

### 3. `features`

- Module for extracting features from preprocessed images.
- Extracts relevant features for font recognition.

### 4. `modeltraining`

- Module for training machine learning models.
- Utilizes extracted features to train a font recognition model.

### 5. `main`

- Main script orchestrating the font recognition process.
- Integrates data reading, preprocessing, feature extraction, model training, testing, and accuracy evaluation.

## Usage

To use the font recognition system, follow these steps:

1. Set the `path_folder` variable to the directory containing the font dataset.
2. Configure the `NumberOftrainingData` and `NumoftestData` variables according to your dataset size.
3. Run the `TrainingModule()` function to train the font recognition model.
4. Run the `ReadTestData()` function to load test data.
5. Run the `TestingModule(testlabels, test)` function to test the trained model and evaluate accuracy.

## Files

- `readdata.py`: Contains functions for reading font image data.
- `preprocessing.py`: Performs image preprocessing tasks.
- `features.py`: Extracts features from preprocessed images.
- `modeltraining.py`: Trains the font recognition model.
- `main.py`: Main script orchestrating the font recognition process.
- `results.txt`: Contains predicted labels for test data.
- `time.txt`: Contains execution times for the testing phase.

## Requirements

- Python 3.x
- Required Python libraries: *joblib, opencv-python, scipy, numpy, scikit-learn*

