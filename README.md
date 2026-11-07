KNN – Heart Disease Prediction

<p align="center">
  <strong>A Machine Learning project using the K-Nearest Neighbors (KNN) algorithm to predict heart disease.</strong>
</p><p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Algorithm-KNN-green" />
  <img src="https://img.shields.io/badge/Project-Classification-purple" />
</p>---

Overview

KNN – Heart Disease Prediction is a Machine Learning project that uses the K-Nearest Neighbors (KNN) algorithm to classify whether a patient has heart disease or not based on medical features.

The project uses the UCI Heart Disease dataset and applies a supervised Machine Learning approach to analyze patient data and make predictions.

KNN works by finding the nearest data points to a new record and predicting its class based on the majority class among its nearest neighbors.

Features

- Heart disease classification using KNN.
- Uses the UCI Heart Disease dataset.
- Binary classification: Heart Disease / No Heart Disease.
- Data preprocessing and preparation.
- Model training and testing.
- Model evaluation using classification metrics.
- Prediction for new patient records.

Technologies Used

Technology| Purpose
Python| Main programming language
Pandas| Data manipulation
NumPy| Numerical operations
Scikit-learn| Machine Learning implementation
UCI Repository| Dataset source

Dataset

The project uses the Heart Disease Dataset from the UCI Machine Learning Repository.

- Dataset Name: Heart Disease
- Dataset ID: 45
- Number of Features: 13
- Original Target: "num"
- Task: Binary Classification

Target Transformation

The original target column contains values from 0 to 4.

For this project, the target is converted into two classes:

Original Value| New Class
0| No Heart Disease
1, 2, 3, 4| Heart Disease

Input Features

The model uses 13 medical features:

Feature| Description
age| Patient's age
sex| Sex value in the dataset
cp| Chest pain type
trestbps| Resting blood pressure
chol| Serum cholesterol
fbs| Fasting blood sugar
restecg| Resting ECG results
thalach| Maximum heart rate achieved
exang| Exercise-induced angina
oldpeak| ST depression
slope| Slope of peak exercise ST segment
ca| Number of major vessels
thal| Thalassemia-related value

Machine Learning Algorithm

K-Nearest Neighbors (KNN)

KNN is a supervised Machine Learning algorithm used for classification and regression.

It predicts the class of a new data point by calculating its distance from the training data and finding the K nearest neighbors.

The final prediction is based on the majority vote of those neighbors.

How KNN Works

1. Choose the number of neighbors (K).
2. Calculate the distance between the new data point and training samples.
3. Find the K nearest neighbors.
4. Check the classes of the selected neighbors.
5. Assign the majority class as the prediction.

Machine Learning Workflow

1. Load the Heart Disease dataset.
2. Explore and understand the dataset.
3. Separate input features and target.
4. Convert the target into binary classes.
5. Handle missing values and prepare the data.
6. Split the dataset into training and testing sets.
7. Train the KNN classifier.
8. Evaluate the model performance.
9. Use the trained model to make predictions.

Model Evaluation

The model performance can be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help measure how well the model classifies the test data.

«Add your actual evaluation results here after running the model.»

Project Structure

KNN-Heart-Disease-Prediction/
│
├── ml_model.py
├── ui.py
├── terminal_test.py
├── README.md
│
└── models/

Update the structure to match the actual files in your project.

Installation

Clone the repository:

git clone YOUR_REPOSITORY_LINK

Install the required libraries:

pip install pandas numpy scikit-learn ucimlrepo

How to Run

Run the Machine Learning model:

python ml_model.py

Run the user interface:

python ui.py

Run the terminal testing script:

python terminal_test.py

Use the commands that match your actual project files and interface.

Important Notes

- KNN is a distance-based algorithm.
- Feature scaling can affect KNN performance.
- Choosing the right value of K is important.
- Data preprocessing is necessary before training.
- Model performance depends on the dataset and preprocessing steps.

Limitations

- The dataset has a limited number of records.
- KNN can be affected by irrelevant or differently scaled features.
- The choice of K can change the prediction results.
- The model's performance on this dataset does not guarantee the same performance on real-world medical data.

Future Improvements

- Test different values of K.
- Compare different distance metrics.
- Improve data preprocessing.
- Apply feature selection.
- Compare KNN with other classification algorithms such as SVM and Decision Tree.

Disclaimer

This project is developed for educational and academic purposes only.

It is not a medical diagnostic tool and should not be used to make real medical decisions.

Acknowledgements

- UCI Machine Learning Repository for providing the Heart Disease dataset.
- Scikit-learn for Machine Learning tools and algorithms.
