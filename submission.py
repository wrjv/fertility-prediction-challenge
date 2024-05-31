"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    # Partner
    df["y_partner"] = (df["cf20m024"] % 2).fillna(0)

    # Married
    df["y_married"] = (df["cf20m030"] % 2).fillna(0)

    # Years living together [YEARS TOGETHER DIDN'T WORK]
    df["y_together"] = 2024 - df["cf20m029"]
    df["y_together"] = df["y_together"].fillna(0)
    df["y_together2"] = df["y_together"] * df["y_together"]
    df["y_together3"] = df["y_together"] * df["y_together2"]

    # Relationship satisfaction [CURRENTLY NOT USED BECAUSE LOGIT BECAME MUCH WORSE]
    # Imputing missing values with the mean
    df["satisfied"] = np.where(df['cf20m180'] == 999, np.NaN, df['cf20m180'])
    df["satisfied"] = df["satisfied"].fillna(df["satisfied"].mean())

    # Age difference between both partners [NOT USED, MAKES PREDICTIONS WORSE]
    df["age_diff_mf"] = np.where((df['cf20m003'] == 2) & (df['cf20m003'] != df['cf20m032']), df["birthyear_bg"] - df['cf20m026'], np.NaN)
    df["age_diff_mf"] = np.where((df['cf20m003'] == 1) & (df['cf20m003'] != df['cf20m032']), df['cf20m026'] - df["birthyear_bg"], df["age_diff_mf"])
    df["age_diff_mf"] = df["age_diff_mf"].fillna(df["age_diff_mf"].mean())
    # FEMALE PARTNER AGE
    # If male, use partners age, if female, use own age
    # Age has range 17 - 70
    # df['age'] = np.where(df['cf20m003'] == 2, 2024 - df['cf20m026'], 2024 - df["birthyear_bg"])
    # df['age'] = np.where((df['cf20m003'] == 2) & (df['age'] == np.NaN), 2024 - df["birthyear_bg"] - df["age_diff_mf"].mean(), df['age'])

    # RESPONDENT AGE
    # Age, Age^2 & Age^3;
    df['age'] = 2024 - df["birthyear_bg"]
    df["age"] = df["age"].fillna(df["age"].mean())
    df["age2"] = df["age"] * df["age"]
    df["age3"] = df["age"] * df["age2"]

    # Want child
    df["child_want"] = np.where((df["cf20m128"] == 1), 1, 0)
    df["child_want_years"] = np.where((df["cf20m130"] <= 1), 1, 0)
    df["child_want_years2"] = np.where((df["cf20m130"] <= 2), 1, 0)
    df["child_want_years3"] = np.where((df["cf20m130"] <= 3), 1, 0)

    df["religion"] = np.where((df["cr20m041"] <= 3), 1, 0)

    # Trouble making ends meet
    df["poor"] = np.where((df["ci20m245"] == 1), 1, 0)

    # Selecting variables for modelling
    keepcols = [
        "nomem_encr",  # ID variable required for predictions
        "age", "age2", "age3",
        "y_together", "y_together2", "y_together3",
        "y_partner",
        "child_want", "child_want_years", "child_want_years2", "child_want_years3",
        "religion", "poor"
    ] 

    # Keeping data with variables selected
    df = df[keepcols]

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict

def main():
    train = pd.read_csv("../data/training_data/PreFer_train_data.csv", low_memory = False) 
    fake = pd.read_csv("../data/other_data/PreFer_fake_data.csv") 
    print(predict_outcomes(train))

if __name__ == '__main__':
    main()
