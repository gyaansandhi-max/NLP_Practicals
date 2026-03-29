# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

# Step 2: Load Dataset
df = pd.read_csv("data.csv")

# -----------------------------------------
#  Handle missing values properly
# -----------------------------------------

# Text columns
text_cols = df.select_dtypes(include='object').columns
df[text_cols] = df[text_cols].fillna("unknown")

# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# -----------------------------------------
# Step 3: Create TEXT column
# -----------------------------------------
df['text'] = (
    df['Make'] + " " +
    df['Model'] + " " +
    df['Engine Fuel Type'] + " " +
    df['Transmission Type'] + " " +
    df['Driven_Wheels'] + " " +
    df['Market Category'] + " " +
    df['Engine Cylinders'].astype(str) + "cyl " +
    df['Engine HP'].astype(str) + "hp " +
    df['Year'].astype(str)
)

# Lowercase for consistency (IMPORTANT)
df['text'] = df['text'].str.lower()

print("Sample Text:\n", df['text'].head())

# -----------------------------------------
#  BAG OF WORDS
# -----------------------------------------
cv = CountVectorizer()
bow = cv.fit_transform(df['text'])

print("\nBag of Words Shape:", bow.shape)

# -----------------------------------------
#  NORMALIZED BAG OF WORDS
# -----------------------------------------
bow_norm = normalize(bow, norm='l1')
print("\nNormalized BoW (first row):")
print(bow_norm.toarray()[0])

# -----------------------------------------
#  TF-IDF
# -----------------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(df['text'])

print("\nTF-IDF Shape:", tfidf.shape)

# -----------------------------------------
#  WORD2VEC
# -----------------------------------------
sentences = [text.split() for text in df['text']]

w2v_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=4)

# Better example word (dataset-based)
print("\nEmbedding for 'premium':")
if 'premium' in w2v_model.wv:
    print(w2v_model.wv['premium'])

# Sentence embedding
def sentence_vector(sentence):
    words = sentence.split()
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(50)

df['embedding'] = df['text'].apply(sentence_vector)

print("\nSample Sentence Embedding:")
print(df['embedding'].iloc[0])