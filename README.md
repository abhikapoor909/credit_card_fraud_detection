# Credit Card Fraud Detection: Handling Imbalanced Data with Machine Learning and Neural Networks

This project implements an end-to-end machine learning pipeline to detect credit card fraud using the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, with fraud cases making up only ~0.17% of transactions, The project compares multiple models, including traditional machine learning algorithms and a neural network, to evaluate their performance on this task.

## Features
- **Data Preprocessing**: Loads, splits, and scales the dataset.
- **Imbalance Handling**: Implements techniques like class weighting and SMOTE (Synthetic Minority Oversampling Technique).
- **Model Comparison**: Evaluates Logistic Regression, Decision Tree, Random Forest, XGBoost, Balanced Random Forest, and a Neural Network.
- **Performance Metrics**: Uses precision, recall, F1-score, and precision-recall curves with average precision (AP) scores to assess models on imbalanced data.
- **Visualization**: Plots precision-recall curves for all models.

## Dataset
The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, containing 284,807 transactions with 31 features (Time, V1-V28, Amount, and Class). Fraudulent transactions (Class=1) are rare, posing a challenge for classification.

