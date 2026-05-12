import pandas as pd
import joblib

# Check cleaned data
df = pd.read_csv('../data/processed/clean_train.csv')
print(df.columns.tolist())
print(df.head(2))
print(df.shape)

# Check encoder
encoder = joblib.load('../models/onehot_encoder.pkl')
print(type(encoder))