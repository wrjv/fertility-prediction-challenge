import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from training import train_save_model
from submission import clean_df, predict_outcomes
import sys

def score(prediction_path, ground_truth_path):
    """Score (evaluate) the predictions and write the metrics.
    """

    # Load predictions and ground truth into dataframes
    predictions_df = pd.read_csv(prediction_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Merge predictions and ground truth on the 'id' column
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")

    # Calculate accuracy
    accuracy = len(merged_df[merged_df["prediction"] == merged_df["new_child"]]) / len(
        merged_df
    )

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame(
        {
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1_score],
        }
    )
    metrics_df.to_csv(sys.stdout, index=False)

def main():
    print("Load data")
    # loading data (predictors)
    train = pd.read_csv("../data/training_data/PreFer_train_data.csv", low_memory = False) 
    # loading the outcome
    outcome = pd.read_csv("../data/training_data/PreFer_train_outcome.csv") 

    print("Clean data")
    train_cleaned = clean_df(train)

    # exit(0)

    print("Train model")
    train_save_model(train_cleaned, outcome)

    predict_outcomes(train).to_csv('../data/train_predictions.csv')

    print("Score model")
    score('../data/train_predictions.csv', '../data/training_data/PreFer_train_outcome.csv')

if __name__ == '__main__':
    main()