import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import joblib

# Function to read the CSV file
def d_read_csv(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath)
    return df

# Function to clean the dataframe
def d_cleanse_df(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(["nctid"], axis=1, inplace=True)
    return df

# Function to preprocess the text
def d_text_processing(data):
    nltk.download('stopwords')
    nltk.download('wordnet')
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words of length 1 or 2
        text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only letters
        text = text.lower()
        text = text.split()
        text = [word for word in text if word not in stop_words]
        text = [wnl.lemmatize(word) for word in text]
        text = ' '.join(text)
        return text
    
    if isinstance(data, pd.DataFrame):
        data['description'] = data['description'].apply(clean_text)
        return data
    elif isinstance(data, str):
        return clean_text(data)

# Function to train and evaluate the model
def d_train_evaluate(df: pd.DataFrame) -> None:
    X = df['description']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(solver='liblinear'))
    ])

    # Define the parameter grid
    param_grid = {
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__min_df': [1, 5],
        'clf__C': [0.1, 1, 10]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, 'best_text_classification_model.pkl')

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report:\n {classification_report(y_test, y_pred)}')
    print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')

# Function to load the saved model
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model

# Function to predict the label for a new description
def predict_description(model, description: str) -> str:
    processed_description = d_text_processing(description)
    prediction = model.predict([processed_description])
    return prediction[0]

# Main function to run the pipeline
def main():
    df = d_read_csv("data/trials.csv")
    clean_df = d_cleanse_df(df)
    pre_processed_df = d_text_processing(clean_df)
    d_train_evaluate(pre_processed_df)

    # Example usage
    model_path = 'best_text_classification_model.pkl'
    loaded_model = load_model(model_path)

    new_description = "this is a test description about Dementia"
    predicted_label = predict_description(loaded_model, new_description)
    print(f'The predicted label for the new description is: {predicted_label}')
