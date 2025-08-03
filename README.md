Credit Card Default Prediction using Machine Learning
Project Overview
This project aims to predict the probability of a credit card client defaulting on their next monthly payment. The analysis uses a publicly available dataset from the UCI Machine Learning Repository.

The primary goal is to build and evaluate a predictive model that can classify clients as either "default" or "not default." This helps in understanding the key factors contributing to default and can be used by financial institutions for risk assessment.

Files in this Repository
Default of Credit Card Clients - Predictive Models.IPynb: This Jupyter Notebook contains the complete data analysis and model building process. It includes data exploration, preprocessing, model training, and evaluation.

UCI_Credit_Card.csv: This is the dataset used for the project. It contains a range of features for credit card clients, including demographic information, credit history, bill amounts, and payment history.

Methodology
The Jupyter Notebook follows a standard machine learning workflow:

Data Loading and Initial Exploration: The UCI_Credit_Card.csv file is loaded into a pandas DataFrame. Basic data exploration is performed to understand the structure, check for missing values, and analyze the distribution of the target variable (default.payment.next.month).

Exploratory Data Analysis (EDA): Various visualizations are created using seaborn and matplotlib to understand the data.

Density plots of credit limit amounts.

Boxplots to visualize the relationship between credit limit, age, and demographic features like sex and marriage status.

Correlation heatmaps to identify relationships between the bill amounts (BILL_AMT) and previous payment amounts (PAY_AMT).

Feature Engineering & Preprocessing:

The raw data is split into training and validation sets.

Categorical features such as EDUCATION, SEX, and MARRIAGE are one-hot encoded (dummified) to make them suitable for the machine learning model.

The PAY_ features are also one-hot encoded to handle their categorical nature effectively.

Model Training:

A RandomForestClassifier is chosen for its strong performance and ability to handle complex datasets.

The model is trained on the preprocessed training data.

Model Evaluation:

The trained model's performance is evaluated on the validation set.

A confusion matrix is generated to show the number of correct and incorrect predictions for each class.

The Area Under the Receiver Operating Characteristic Curve (ROC AUC) score is calculated to measure the model's overall discriminative power. A higher AUC score indicates a better model.

Feature importance is plotted to identify which features had the most significant impact on the model's predictions.

Key Findings & Results
Most Important Features: Based on the Random Forest model, the most important features for predicting default are the PAY_ features (repayment status), LIMIT_BAL (credit limit), AGE, and BILL_AMT (bill amounts).

Model Performance: The model achieved an ROC AUC score of approximately 0.66, which is a good starting point for a predictive model on this dataset.

One-Hot Encoding: Using one-hot encoded categorical features improved the model's performance, although the increase in the AUC score was small.

How to Run the Project
To run this project, you will need a Python environment with the following libraries installed:

pandas

numpy

scikit-learn

seaborn

matplotlib

You can install them using pip:

pip install pandas numpy scikit-learn seaborn matplotlib

After installing the libraries, you can open and run the Jupyter Notebook Default of Credit Card Clients - Predictive Models.IPynb in a Jupyter or Colab enviro
