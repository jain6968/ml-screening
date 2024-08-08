# RGrid Machine Learning Challenge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Here is my solution!

## Situation/ Objective: 
> As understood the data is medical data to clasify label into 5 types mainly ALS, Demetia, Obessive Compulsive Disorder, Parkinson's Disease, Scoliosis

> First I setup the configurations and requirements for the code, and then created a file file_text_analysis.py for all the core work. 

Main objective as understood is to create best model for predicting the output of the description into those 5 categroies.

## Task: 
My main task is to first understand, evaluate the data, clean, and pre-process.
Next task is to feed into various models and evaluate the accuracy, precision, recall, F-score and if required iterate to get best result. Store the model. Call it as an api in the main.py for any description given to the function.  

# Action:
I mainly used nltk, sklearn with basic pandas to get the job done. 
I mainly did the following task,

1. Data Handling:
Reads data from a CSV file.
Cleans the data by removing unnecessary columns.

2. Text Preprocessing:

Cleans up the text by removing irrelevant characters, stopwords, and reducing words to their base form (lemmatization).

3.Model Training:
Splits the data into training and testing sets.
Converts text data into numerical features using TF-IDF (which turns text into numbers that represent how important each word is in the text).
Trains a Logistic Regression model to predict labels based on the text.

4.Model Tuning and Evaluation:
Uses GridSearchCV to find the best hyperparameters (settings) for the model.
Evaluates the model's performance using accuracy and other metrics.
Saves the best-performing model.


> Why GridSearchCV: 
GridSearchCV was used to automatically search for the best combination of hyperparameters (like regularization strength) for the Logistic Regression model. Instead of manually trying different settings, GridSearchCV tries them all and finds the one that gives the best performance, making the model more accurate and effective.