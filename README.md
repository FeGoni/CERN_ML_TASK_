## CERN ML Task

## Project Overview
This project implements a complete Machine Learning pipeline to predict customer subscription behavior. The goal is to classify customers into two groups: those likely to subscribe (Target = 1) and those who are not (Target = 0).
## Assignment Details
⦁	Assigned Model: Linear Discriminant Analysis (LDA) (day of birth % 5 == 0)
⦁	Dataset: Customer demographic and behavioral data (including HH Income, Age, Nielsen Prizm segments, Source Channel, etc.) (339034 % 9 == 4)
## Workflow & Methodology
The project follows a structured 4-step pipeline:
## 1. Data Analysis and Cleaning
⦁	Categorical Normalization: Cleaned values in Source Channel (reduced 50+ variations to 7 logical groups like Digital, DirectSales, Phone) and Delivery period. This step allowed us to retain more data instead of dropping rows.
⦁	Numeric Conversion: Parsed complex string intervals in HH Income, Age, and Weekly Fee.
⦁	Missing Values: Addressed missing data (e.g., creating a "Other/NoData" group for Language).
## 2. Feature Engineering
New features were created to capture deeper relationships in the data:
⦁	Income Spent on subscription (%): Created a new variable to help our LDA model.
⦁	Prizm_Year & Prizm_Knowledge: Decoded the Nielsen Prizm column into two separate semantic features (Lifestage and Knowledge).
⦁	Dimensionality Reduction: Applied One-Hot Encoding with drop_first=True to prevent multicollinearity.
## 3. Model Optimization
⦁	Train/Test Split: Performed a stratified split (80/20) to preserve class distribution.
⦁	Preprocessing: Applied StandardScaler to normalize features.
⦁	Hyperparameter Tuning: Used GridSearchCV to optimize the LDA model parameters.
## 4. Final Validation
The tuned model was evaluated on the held-out test set (20% of total data).
## Conclusion
The analysis highlights the limitations of using Linear Discriminant Analysis for this specific dataset. LDA relies on following a Gaussian distribution. This assumption clashes with the nature of our data, where the strongest predictors are categorical (Source Channel, Nielsen Prizm) and sparse after One-Hot Encoding. Therefore the model struggles to find an optimal linear boundary. We achieved the result of 58% recall and 45% f1-score.
