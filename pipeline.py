import streamlit as st
import pandas as pd
import re
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from scipy.sparse import hstack
from wordcloud import WordCloud

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

st.set_page_config(page_title="Fake Review Detection", layout="wide")

st.title("🛒 Fake Product Review Detection System")

st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "SVM", "Random Forest"]
)

max_features = st.sidebar.slider("TF-IDF Features", 1000, 10000, 5000)
C_value = st.sidebar.slider("Logistic Regularization", 0.1, 5.0, 1.0)
min_length = st.sidebar.slider("Min Review Length", 0, 500, 10)

st.subheader("📂 Upload Dataset")
file = st.file_uploader("Upload CSV")

if file:

    try:
        df = pd.read_csv(file, encoding='latin-1')
    except:
        st.error("❌ Failed to read CSV")
        st.stop()

    df.columns = [c.lower().strip() for c in df.columns]

    if 'review_text' not in df.columns or 'label' not in df.columns:
        st.error("❌ Dataset must contain 'review_text' and 'label'")
        st.stop()

    df.dropna(subset=['review_text'], inplace=True)

    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df['label'] = df['label'].map({"OR": 0, "CG": 1, "0": 0, "1": 1})
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    st.write("Label Distribution:", df['label'].value_counts())

    if df['label'].nunique() < 2:
        st.error("❌ Dataset must contain both classes")
        st.stop()

    df['cleaned'] = df['review_text'].apply(clean_text)
    df['length'] = df['review_text'].apply(len)

    df = df[df['length'] >= min_length]

    Q1 = df['length'].quantile(0.25)
    Q3 = df['length'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['length'] >= Q1 - 1.5*IQR) & (df['length'] <= Q3 + 1.5*IQR)]

    df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
    df['exclamation_count'] = df['review_text'].apply(lambda x: str(x).count('!'))
    df['caps_ratio'] = df['review_text'].apply(lambda x: sum(1 for c in str(x) if c.isupper())/(len(str(x))+1))

    keywords = ["best ever", "life changing", "must buy", "100%", "amazing amazing", "best product ever"]
    df['exaggeration'] = df['review_text'].apply(lambda x: any(k in str(x).lower() for k in keywords)).astype(float)

    df['repetition'] = df['review_text'].apply(lambda x: len(str(x).lower().split()) - len(set(str(x).lower().split())))

    majority = df[df.label == 0]
    minority = df[df.label == 1]

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df = pd.concat([majority, minority_upsampled])

    st.write("Balanced Distribution:", df['label'].value_counts())

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    text_features = vectorizer.fit_transform(df['cleaned'])

    extra_features = df[['word_count', 'exclamation_count', 'caps_ratio', 'exaggeration', 'repetition']].astype(float).values

    X = hstack([text_features, extra_features])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(C=C_value, class_weight='balanced', max_iter=1000)
    elif model_choice == "SVM":
        model = SVC(probability=True, class_weight='balanced')
    else:
        model = RandomForestClassifier(class_weight='balanced')

    model.fit(X_train, y_train)

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model", "📝 Predict"])

    with tab3:
        st.subheader("📝 Test Your Review")

        user_input = st.text_area("Enter product review:")

        if st.button("🔍 Analyze Review"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a review")
            else:
                cleaned = clean_text(user_input)
                text_vec = vectorizer.transform([cleaned])

                wc = float(len(user_input.split()))
                ex = float(user_input.count('!'))
                caps = float(sum(1 for c in user_input if c.isupper())/(len(user_input)+1))
                exaggeration = float(any(k in user_input.lower() for k in keywords))
                repetition = float(len(user_input.lower().split()) - len(set(user_input.lower().split())))

                extra = [[wc, ex, caps, exaggeration, repetition]]
                final_vec = hstack([text_vec, extra])

                pred = model.predict(final_vec)[0]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(final_vec)[0][1]
                else:
                    prob = 0.5

                st.progress(prob)
                st.write(f"Fake Probability: {prob*100:.2f}%")

                if pred == 1:
                    st.error("🚨 Fake Review Detected")
                else:
                    st.success("✅ Genuine Review")

else:
    st.info("⬆️ Upload dataset to start")
