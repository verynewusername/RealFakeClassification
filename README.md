# RealFakeClassification

Code used in my Bachelor Thesis.

## Overview

This project focuses on the classification of real and fake images. The goal is to develop and evaluate a machine learning model that can effectively distinguish between genuine and manipulated images.

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks used for data exploration and model development.
- `src/`: Source code for the machine learning models and utility functions.
- `models/`: Pre-trained models and saved checkpoints.
- `results/`: Results and evaluation metrics of the models.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/verynewusername/RealFakeClassification.git
cd RealFakeClassification
pip install -r requirements.txt
```

## Usage

To train the model, run:

```bash
python src/train.py
```

To evaluate the model, run:

```bash
python src/evaluate.py
```

## Results

The results of the classification models are stored in the `results/` directory. Detailed evaluation metrics and visualizations can be found in the notebooks within the `notebooks/` directory.
