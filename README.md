
# Audio CAPTCHA AI Model Trainer and Tester

This repository contains Python scripts for training and testing an AI model for audio CAPTCHA recognition. The main components are:

- `CaptchaTraining.py`: Contains the core functionalities for data handling, model training, and testing.
- `captcha_train.py`: A script to train the model with specified parameters.
- `captcha_test.py`: A script for testing the model on new data.

## Requirements

- Python 3.x
- Libraries: `numpy`, `h5py`, `sklearn`, `scipy`, `pickle`, `joblib`

(You may need to install these libraries using `pip install` if they are not already installed.)

## Training the Model

To train the model, use the `captcha_train.py` script. This script requires a dataset in `.mat` format. 

Get the dataset (too big to upload on Github): https://drive.google.com/file/d/13kybvVMcrj3gRRGgYEGnRlyLpmAH591z/view?usp=sharing

### Usage

Run the script from the command line with the following syntax:

```
python captcha_train.py [train_features_file] [cost_value] [pcavar]
```

- `train_features_file`: Path to the `.mat` file containing training features (default: `../dataset/train/train_features.mat`).
- `cost_value`: Regularization parameter for the SVM classifier (default: `100.0`).
- `pcavar`: Threshold for the cumulative variance in PCA (default: `0.95`).

Example:

```
python captcha_train.py dataset/train/train_features.mat 100.0 0.95
```

This command will train the model using the specified dataset and parameters.

### Training Results

- **Number of Data Read**: Indicates the total number of data points used in training.
- **Success Rate per Class**: Shows the proportion of correctly predicted instances for each class (0-9 and non-numbered parts).

## Testing the Model

The `captcha_test.py` script is used to test the trained model.

### Usage

Run the script from the command line with the following syntax:
```
python captcha_test.py [trained_model_file]
```

`trained_model_file`: Path to the .joblib file containing training features (default: `model_100_0.95.joblib`).

### Testing Results

- **File Name**: Name of the tested audio file (digits pronounced in the .wav file).
- **Predicted Digits**: Digits predicted by the model.
- **Precision**: Overall accuracy of the model, the ratio of correctly predicted files to the total number of files tested.
- **Digit Precision**: Accuracy at the digit level, the ratio of correctly predicted digits to the total number of digits in all tested files.
- **General Precision and Digit Precision**: At the end of testing, the script will output the general precision (overall accuracy of the model) and digit precision (accuracy of digit predictions) across all tested files.


## IMPORTANT

- Scripts may take some time (several minutes for a powerful machine) to run.


## Additional Information

- The `CaptchaTraining.py` script includes detailed documentation and definitions for various functions and methodologies used in the model, such as Cepstral coefficients, PCA, and SVM classifiers.
- Ensure that the dataset and file paths are correctly set according to your directory structure.
