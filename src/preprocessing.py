import pandas as pd
import string
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse

# Define paths based on project structure
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
MODELS_DIR = "../models/"

def clean_text(text):
    """Lowercases text and removes punctuation."""

    if not isinstance(text, str):
        return "" # Handle any missing/NaN values gracefully
    
    # Lowercasing
    text = text.lower()
    
    # Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))    # This function creates a translation table (a map for the computer)
    return text

def preprocess_dataset(filename):
    """Loads a dataset, cleans specific columns, and returns the DataFrame."""

    filepath = os.path.join(RAW_DATA_DIR, filename)
    print(f"Processing {filename}...")
    df = pd.read_csv(filepath)
    
    # Columns required by the schema
    cols_to_clean = ['article', 'question', 'A', 'B', 'C', 'D']
    
    for col in cols_to_clean:
        # Created new cleaned columns to preserve the original text for the UI later
        df[f'clean_{col}'] = df[col].apply(clean_text)
        
    return df

def main():
    # Make output directories if they don't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Process all dataset splits
    train_df = preprocess_dataset('train.csv')
    dev_df = preprocess_dataset('dev.csv')
    test_df = preprocess_dataset('test.csv')

    # Save the cleaned datasets to use
    print("Saving cleaned datasets to data/processed/...")
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clean_train.csv'), index=False)
    dev_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clean_dev.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clean_test.csv'), index=False)

    # Initialize One-Hot Encoding (Primary Feature Representation)
    print("Fitting One-Hot Encoder on training vocabulary...")
    
    # Combine clean articles and questions to build a comprehensive vocabulary
    corpus = train_df['clean_article'].tolist() + train_df['clean_question'].tolist()
    
    # Note: binary=True forces One-Hot Encoding instead of word counts.
    vectorizer = CountVectorizer(binary=True, max_features=5000, stop_words='english')
    vectorizer.fit(corpus)

    # Transform and save feature matrices for Model A/B training
    train_features = vectorizer.transform(train_df['clean_article'] + ' ' + train_df['clean_question'])
    scipy.sparse.save_npz(os.path.join(PROCESSED_DATA_DIR, 'train_features.npz'), train_features)

    # Save the trained encoder so Model A and Model B can transform text later
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'onehot_encoder.pkl'))
    print("Preprocessing complete! Data saved to data/processed/ and Encoder saved to models/")

if __name__ == "__main__":
    main()