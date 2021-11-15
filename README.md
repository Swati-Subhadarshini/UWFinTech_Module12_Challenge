# UWFinTech_Module12_Challenge

This project is designed to evaluate healthy loans and risky loans using supervised learning. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.


---
## Technologies Used

Leveraging python version 3.9.6
Git Bash CLI

# Libraries Used

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# Steps for the Challenge:

1. Split the Data into Training and Testing Sets

2. Create a Logistic Regression Model with the Original Data

3. Predict a Logistic Regression Model with Resampled Training Data

4. Write a Credit Risk Analysis Report

### Split the Data into Training and Testing Sets:

Open the credit_risk_resampling.ipynb file, import the above mentioned libraries and then read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

NOTE
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Check the balance of the labels variable (y) by using the value_counts function.

Split the data into training and testing datasets by using train_test_split.

Create a Logistic Regression Model with the Original Data
Employ your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Predict a Logistic Regression Model with Resampled Training Data

Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use RandomOverSampler.

To do so, complete the following steps:

Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Write a Credit Risk Analysis Report
For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, and make sure that it contains the following:

An overview of the analysis: Explain the purpose of this analysis.

The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation, if any, for the model to use the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.


# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

### Explain the purpose of the analysis:
        This challenge is on using various techniques to train and evaluate models with imbalanced classes. The purpose of this analysis is to use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
        
### Explain what financial information the data was on, and what you needed to predict.
        This financial data was on historical lending activity from a peer-to-peer lending service company. The data contains various parameter like loan_size, interest_rate, debt values and other. The parameter loan_status is either label 0 which indicates healthy loan and label 1 indicating high-risk loan.

### Write a Credit Risk Analysis Report
        The credit risk analysis report is to make prediction on original data and resampled data (because of the imbalanced class) and compare about the accuracy of the prediction, precision and recall value and compare the accuracy rate of the predictions. The resampling of data helps to make better prediction without being biased on the data set.
        
        
### Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

        value_counts(): Which helps to count the no of values present in each label data.
        
        train_test_split(): This function helps on training testing the spliting the data set before modeling, fiting and predicting the future data.
        
        balanced_accuracy_score(): This function helps in getting accuracy score for the test results (one from test data spliting and prediction of data)
        
        confusion_matrix(): The confusion matrix gives the output of no. of True Positive(TP), True Negative(TN), False Positive(FP), FN(False Negative).
        
        classification_report_imbalanced(): This function gives us the precision value(Percentage of prediction were correct), Recall value (Fraction of             positives that were correctly identified), F1 Score (weighted harmonic mean of precisions and recall such that the best score is 1 and worst is 0.0)
        
        RandomOverSampler(): This function is used for oversample the data to have a balanced class and prediction can be done without biased.
        
        fit_resample(): This function is used to fit the sampled data into the model.
        
        
### Describe the stages of the machine learning process you went through as part of this analysis.

        Split the Data into Training and Testing Sets

        Create a Logistic Regression Model with the Original Data

        Predict a Logistic Regression Model with Resampled Training Data

        Write a Credit Risk Analysis Report
       

### Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

     Logistic Regression is used fot the methods and RandomOverSampler has been used for resampling method for prediction and comparision of better output in terms of accuracy, precision, recall, F1 score value (using classification report function)


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
        The accuracy score for the logistic regressio model is (0.95 approximately) and with avg precision precision value of 0.99 and recall value (0.99).


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
        The accuracy score for oversampled data is (0.99) meaning that the model using resampled data was much better at detecting true positives and true           negatives. The precision with the resampled data (0.99).

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

As per the analysis of prediction on original data and resampling data the precision and recall value holds 0.99 which same for both the cases. However the accuracy is better after resampling of data aroung 99% comparing to the original data prediction of 95%. 

Performance depends on the predicted value of loan status where the 1's are high risk loans and 0's are healthy loans.
In the original data set 
0    75036
1     2500

In the Oversampled data 
0    56271
1    56271

In the original data the difference is huge betweeen 0's and 1's so the spliting data and then traing and testing is quite biased however after resampling data into equal values the model gives better result without get biased.

Recommendation would be the oversampled model for the which performed better comparing to the original model.

## Contributors

This project is designed by Swati Subhadarshini 
Emaid id: sereneswati@gmail.com
LinkedIn link: https://www.linkedin.com/in/swati-subhadarshini