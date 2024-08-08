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

3. Model Training:
    Splits the data into training and testing sets.
    Converts text data into numerical features using TF-IDF (which turns text into numbers that represent how important each word is in the text).
    Trains a Logistic Regression model to predict labels based on the text.

4. Model Tuning and Evaluation:
    Uses GridSearchCV to find the best hyperparameters (settings) for the model.
    Evaluates the model's performance using accuracy and other metrics.
    Saves the best-performing model.


5. I used GridSearchCV, and here is the explaination for it. 

The dataset contains medical trial descriptions associated with specific labels (like "ALS"). Here's why GridSearchCV is particularly useful for optimizing a model on this dataset:

1. Complexity of Medical Text Descriptions:
   - Specialized Vocabulary: Medical descriptions often contain complex terminology, which can lead to high-dimensional feature spaces when converted to numerical data. GridSearchCV helps by tuning the parameters of the vectorization process (e.g., TF-IDF), ensuring that the model captures the most important features while ignoring noise.

2. Variability in Text Length and Content:
   - Diverse Sentence Structures: The descriptions in your dataset likely vary in length and structure. Some descriptions might be concise, while others are detailed. GridSearchCV can optimize how the text is tokenized and how n-grams (combinations of words) are used, leading to better feature extraction for such diverse text.

3. Class Imbalance:
   - Imbalanced Labels: If your dataset has an unequal distribution of labels (e.g., more descriptions for ALS compared to other conditions), certain models might struggle with prediction accuracy. GridSearchCV helps find the best regularization parameters that can handle this imbalance, ensuring the model doesn't just predict the majority class.

4. Optimization of Feature Extraction:
   - Tuning TF-IDF Parameters: The effectiveness of the TF-IDF vectorization depends on parameters like `max_df` (maximum document frequency) and `ngram_range`. GridSearchCV can fine-tune these parameters, helping the model to focus on relevant medical terms and ignore common but uninformative words.

5. Precision vs. Recall in Medical Applications:
   - Balancing Trade-offs: In medical text classification, it's crucial to balance precision (correctly identifying a condition) and recall (identifying all relevant cases). GridSearchCV allows you to specify which metric to prioritize, optimizing the model according to the critical needs of medical decision-making.

6. Cross-Validation for Generalization:
   - Ensuring Robustness: Given the limited size of your dataset (1,759 entries), it's essential to ensure that the model generalizes well to unseen data. GridSearchCV uses cross-validation, meaning it repeatedly trains the model on different subsets of the data, ensuring the chosen hyperparameters work well across the entire dataset.



Finally to run and test, please use main.py to run in the terminal to see the output at http://127.0.0.1:5000/predict