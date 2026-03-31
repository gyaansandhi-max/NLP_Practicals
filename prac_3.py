# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Step 2: Load Dataset
df = pd.read_pickle("News_dataset.pickle")

print(df.columns)
print(df.head())

# Step 3: Text Cleaning
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\r|\n', ' ', text)  # remove new lines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df['clean_text'] = df['Content'].apply(clean_text)

# Step 4: Stopwords Removal + Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words]  # lemmatization
    return " ".join(words)

df['processed_text'] = df['clean_text'].apply(preprocess)

# Step 5: Label Encoding
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Category'])

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label_encoded'], test_size=0.15, random_state=42
)

# Step 7: TF-IDF
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Train Shape:", X_train_tfidf.shape)
print("Test Shape:", X_test_tfidf.shape)

# Step 8: Save Outputs
df.to_csv("processed_data.csv", index=False)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train_tfidf, f)

with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test_tfidf, f)

print("All files saved successfully!")