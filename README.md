### Churn Prediction Using Artificial Neural Networks (ANN)

## Overview

This project is a churn prediction model built using a Deep Learning Artificial Neural Network (ANN). Churn prediction is crucial for businesses to understand customer behavior and identify which customers are likely to stop using a product or service. By predicting churn, companies can take proactive measures to retain customers, improving customer loyalty and profitability.

## Project Structure

- *data/*: Contains the dataset used for training and testing the model.
- *notebooks/*: Jupyter notebooks for data exploration, preprocessing, and model training.
- *models/*: Directory to save trained models.
- *scripts/*: Python scripts for preprocessing data, training the model, and evaluating performance.
- *results/*: Directory to store evaluation metrics and plots.
- *README.md*: Overview of the project (this file).

## Features

- *Data Preprocessing*: The dataset is cleaned and preprocessed to handle missing values, categorical variables, and scaling numerical features.
- *Model Architecture*: A deep learning model with multiple dense layers is used to predict churn. The architecture can be easily adjusted by modifying the model's configuration in the script.
- *Training*: The model is trained using the training dataset with appropriate validation.
- *Evaluation*: The model is evaluated using various metrics like accuracy, precision, recall, F1-score, and AUC-ROC curve.
- *Hyperparameter Tuning*: Techniques like Grid Search or Random Search can be applied to find the best hyperparameters.
- *Deployment (Optional)*: The trained model can be deployed using frameworks like Flask or FastAPI for real-time churn prediction.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Getting Started

1. *Clone the repository:*

   bash
   git clone https://github.com/Ayush-Rawat-1/Churn-Modelling.git
   cd churn-ann
   

2. *Install the required packages*
   
3. *Run the Jupyter notebook:*

   bash
   jupyter notebook notebooks/churn_modeling.ipynb
   

   This will guide you through the data exploration, preprocessing, model training, and evaluation steps.

4. *Training the Model:*

   If you prefer running the training as a script, execute the following:

   bash
   python scripts/churn_modelling.py
   

## Dataset

The dataset used for this project includes various features that may influence customer churn, such as:

- Customer demographics (age, gender, etc.)
- Account information (subscription length, payment methods, etc.)
- Usage data (frequency of use, last interaction, etc.)

You can find the dataset in the data/ directory. If you want to use a different dataset, make sure it is properly formatted and update the preprocessing steps accordingly.

## Results

The results of the model 86% acuuracy
