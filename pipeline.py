import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

df = pd.read_csv("your_dataset.csv")

def extra_features(text):
    text = str(text)
    length = len(text)
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    words = text.lower().split()
    repetition = len(words) - len(set(words))
    return [length, exclamations, capitals, repetition]

extra = np.array([extra_features(t) for t in df['review']])

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_text = tfidf.fit_transform(df['review'])

X = hstack([X_text, extra])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_review(text):
    text_vec = tfidf.transform([text])
    extra_vec = np.array([extra_features(text)])
    final_input = hstack([text_vec, extra_vec])
    prob = model.predict_proba(final_input)[0][1]
    prediction = model.predict(final_input)[0]
    return prediction, prob

test = "This is the BEST product EVER!!! Totally amazing!!! Must buy!!!"
pred, prob = predict_review(test)

print("Prediction:", "Fake" if pred==1 else "Genuine")
print("Fake Probability:", prob)
