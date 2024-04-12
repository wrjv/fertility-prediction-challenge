import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from training import train_save_model
from submission import clean_df

# loading data (predictors)
train = pd.read_csv("../data/training_data/PreFer_train_data.csv", low_memory = False) 
# loading the outcome
outcome = pd.read_csv("../data/training_data/PreFer_train_outcome.csv") 

print("Clean data")
train_cleaned = clean_df(train)

print("Train model")
train_save_model(train_cleaned, outcome)