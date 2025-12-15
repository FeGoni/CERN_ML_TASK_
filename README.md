## CERN ML Task

## Project Overview
This project implements a complete Machine Learning pipeline to predict customer subscription behavior. The goal is to classify customers into two groups: those likely to subscribe (Target = 1) and those who are not (Target = 0).

## Assignment Details
⦁	Assigned Model: Linear Discriminant Analysis (LDA) (day of birth % 5 == 0)

⦁	Dataset: https://www.openml.org/search?type=data&sort=runs&id=44226&status=active (339034 % 9 == 4)

## Workflow & Methodology

## 1. Data Analysis and Cleaning

⦁	Cleaned values in Source Channel (reduced 50+ variations to 7 logical groups) and Delivery period. This step allowed us to retain more data instead of dropping rows.

⦁	Parsed strings in HH Income, Age, and Weekly Fee.

⦁	Addressed missing data (e.g. creating a "Other/NoData" group for Language).

## 2. Feature Engineering
New features were created to help the model in classification:

⦁	Income Spent on subscription (%): Created a new variable to help our LDA model.

⦁	Prizm_Year & Prizm_Knowledge: Decoded the Nielsen Prizm column into two separate features (Lifestage and Knowledge).

## 3. Model Optimization

⦁	Train/Test Split: Performed a stratified split (80/20) to preserve class distribution.

⦁	Preprocessing: Tested several scalers to find the best performing one (StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer)

⦁	Hyperparameter Tuning: Used GridSearchCV to optimize the LDA model parameters.

## 4. Final Validation
The tuned model was evaluated on the held-out test set (20% of total data).

## Conclusion
The analysis highlights the limitations of using Linear Discriminant Analysis for this specific dataset. LDA relies on following a Gaussian distribution. This assumption clashes with our data, where the strongest predictors are categorical (Source Channel, Nielsen Prizm) and sparse after One-Hot Encoding. Therefore the model struggles to find an optimal linear boundary. We achieved the result of 57% recall and 47% f1-score. Our model struggles with predicting subscribers due to the high imbalance in our dataset, which can be seen on the confusion matrix.
