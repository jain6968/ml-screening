from flask import Flask, jsonify, request
from typing import Literal
from file_text_analysis import *


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]


def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """
    predicted_label = d_train_evaluate(description)
    
    return predicted_label


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["GET","POST"])
def identify_condition():
    #data = request.get_json(force=True)
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
    return predicted_label
    

if __name__ == "__main__":
    app.run()