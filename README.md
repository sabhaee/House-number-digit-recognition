# Street View House Number Detection and Classification

This project focuses on detecting and classifying house numbers using Convolutional Neural Networks (CNNs). The goal is to accurately recognize and classify digits present in images of house numbers.

## Project Structure

The project structure is organized as follows:

- `environment.yml`: This file contains the libraries and dependencies required to set up the project environment.
- `inference.py`: This script perfoms the detection and recognition on the input images.
- `data_prep.py`: This script is responsible for creating the dataset based on the Street View House Numbers (SVHN) dataset. It requires downloading the .mat files from the SVHN website.
- `classifier.py`: This file contains the CNN model and dataloader needed for model inference.
- `train.py`: This script is used for running the training and testing process.
- `detect_and_classify.py`: This is the main code used to process images, detect regions of interest, and classify the digits.
- `/input_images`: This directory contains the input images.
- `/output`: This directory is where the detected and annotated images will be saved.
- `/checkpoints`: This directory should contain the .pth file for the saved model weights. The weights can be downloaded from the provided link (`best_svhn_model_state_MyModel_weights.pth`).
- `/plots`: This directory contains plots generated during the training process.
- `/data`: This directory should contain the training and testing data. The data can be downloaded from the provided link.

## Usage

To use this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/sabhaee/House-number-digit-recognition.git
cd house-number-detection
```
2. Set up the project environment by installing the required libraries and dependencies listed in environment.yml. Run the following command:
```bash
conda env create -f environment.yml
```
3. Activate the created environment:
```bash
conda activate house-number-detection
```
4. Place the input photos in the input folder.
5. Use the run.py script to process images, detect regions of interest, and classify the digits:
```bash
python inference.py
```
6. Check the `output` folder for the results.
## Acknowledgments
This project is based on the Street View House Numbers (SVHN) dataset. The weights for the trained model can be downloaded from the provided link.