# Arabic Font Recognition System 🖋️📚

## 📙 Overview

This project is an Arabic font recognition system implemented in Python. It consists of modules for data reading, preprocessing, feature extraction, model training, testing, and accuracy evaluation. The system is designed to recognize fonts from images and provide accuracy metrics based on the trained model, achieving an impressive 99% accuracy. 🌟

## Table of Contents 📑

- <a href="#overview">📙 Overview</a>
- <a href="#modules">🔧 Modules</a>
    - <a href="#1-readdata">1.  📂`readdata`</a>
    - <a href="#2-preprocessing">2. 🧹`preprocessing` </a>
    - <a href="#3-features">3. 🔍 `features`</a>
    - <a href="#4-modeltraining">4. 🏋️`modeltraining` </a>
    - <a href="#5-main">5.🎬 `main` </a>
- <a href="#usage">🚀 Usage</a>
- <a href="#files">🗂️ Files</a>
- <a href="#requirements">🛠️ Requirements</a>
- <a href="#accuracy">🎯 Accuracy</a>
- <a href="#contributors">✨ Contributors</a>
<hr style="background-color: #4b4c60"></hr>

## 🔧 Modules 

### 1.  📂 `readdata`

- Module for reading font images and labels from specified directories.
- Provides functions for reading data for training and testing.

### 2. 🧹 `preprocessing` 

- Module for preprocessing raw image data.
- Performs tasks such as resizing, normalization, and noise reduction.

### 3. 🔍`features` 

- Module for extracting features from preprocessed images.
- Extracts relevant features for font recognition.

### 4. 🏋️`modeltraining`

- Module for training machine learning models.
- Utilizes extracted features to train a font recognition model.

### 5.🎬 `main` 

- Main script orchestrating the font recognition process.
- Integrates data reading, preprocessing, feature extraction, model training, testing, and accuracy evaluation.

## 🚀Usage 

To use the font recognition system, follow these steps:

1. Set the `path_folder` variable to the directory containing the font dataset.
2. Configure the `NumberOftrainingData` and `NumoftestData` variables according to your dataset size.
3. Run the `TrainingModule()` function to train the font recognition model.
4. Run the `ReadTestData()` function to load test data.
5. Run the `TestingModule()` function to test the trained model and evaluate accuracy.

## 🗂️ Files 

- `readdata.py`: Contains functions for reading font image data.
- `preprocessing.py`: Performs image preprocessing tasks.
- `features.py`: Extracts features from preprocessed images.
- `modeltraining.py`: Trains the font recognition model.
- `main.py`: Main script orchestrating the font recognition process.
- `results.txt`: Contains predicted labels for test data.
- `time.txt`: Contains execution times for the testing phase.

## 🛠️ Requirements 

- Python 3.x
- Required Python libraries: *joblib, opencv-python, scipy, numpy, scikit-learn*

## 🎯Accuracy 

Our system has achieved an outstanding 99% accuracy in recognizing Arabic fonts! 🏆

## <img  align="center" width= 70px height =55px src="https://media0.giphy.com/media/Xy702eMOiGGPzk4Zkd/giphy.gif?cid=ecf05e475vmf48k83bvzye3w2m2xl03iyem3tkuw2krpkb7k&rid=giphy.gif&ct=s"> Contributors  

<table align="center" >
  <tr>
      <td align="center"><a href="https://github.com/SH8664"><img src="https://avatars.githubusercontent.com/u/113303945?v=4" width="150px;" alt=""/><br /><sub><b>Sara Bisheer</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/rawanMostafa08"><img src="https://avatars.githubusercontent.com/u/97397431?v=4" width="150px;" alt=""/><br /><sub><b>Rawan Mostafa</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com//mennamohamed0207"><img src="https://avatars.githubusercontent.com/u/90017398?v=4" width="150px;" alt=""/><br /><sub><b>Menna Mohammed</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/fatmaebrahim"><img src="https://avatars.githubusercontent.com/u/113191710?v=4" width="150;" alt=""/><br /><sub><b>Fatma Ebrahim</b></sub></a><br /></td>
  </tr>
</table>
