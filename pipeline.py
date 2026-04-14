import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from wordcloud import WordCloud

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="Fake Review Detection", layout="wide")
st.title("🛒 Fake Product Review Detection System")

# -------------------------------
# SIDEBAR OPTIONS
# -------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "SVM", "Random Forest"]
)

k_fold = st.sidebar.slider("K-Fold Value", 2, 10, 5)

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("📂 Upload Dataset")

file = st.file_uploader("Upload CSV file (must contain 'review_text' and 'label')")

if file:
    df = pd.read_csv(file, encoding='latin-1')

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # DATA CLEANING
    # -------------------------------
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['cleaned'] = df['review_text'].apply(clean_text)

    # -------------------------------
    # EDA SECTION
    # -------------------------------
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Label Distribution")
        st.bar_chart(df['label'].value_counts())

    with col2:
        st.write("Review Length Distribution")
        df['length'] = df['review_text'].apply(len)
        st.bar_chart(df['length'])

    # WORD CLOUD
    st.subheader("☁️ Word Cloud")
    text_data = " ".join(df['cleaned'])

    wordcloud = WordCloud(width=800, height=400).generate(text_data)
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    st.subheader("🔧 Feature Engineering (TF-IDF)")

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    st.write("Feature Matrix Shape:", X.shape)

    # -------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # MODEL SELECTION
    # -------------------------------
    st.subheader("🤖 Model Training")

    if model_choice == "Logistic Regression":
        model = LogisticRegression()

    elif model_choice == "SVM":
        model = SVC()

    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # METRICS
    # -------------------------------
    st.subheader("📈 Performance Metrics")

    try:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    except:
    st.error("⚠️ Error in metrics calculation. Check dataset balance.")
    acc, prec, rec, f1 = 0, 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # CONFUSION MATRIX
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

    # -------------------------------
    # HYPERPARAMETER TUNING
    # -------------------------------
    st.subheader("⚡ Hyperparameter Tuning")

    if st.button("Run Grid Search"):
        if model_choice == "Logistic Regression":
            params = {"C": [0.1, 1, 10]}
            grid = GridSearchCV(LogisticRegression(), params, cv=k_fold)

        elif model_choice == "SVM":
            params = {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
            grid = GridSearchCV(SVC(), params, cv=k_fold)

        else:
            params = {"n_estimators": [50, 100]}
            grid = GridSearchCV(RandomForestClassifier(), params, cv=k_fold)

        grid.fit(X_train, y_train)

        st.write("Best Parameters:", grid.best_params_)
        st.write("Best Score:", grid.best_score_)

    # -------------------------------
    # MANUAL REVIEW CHECK
    # -------------------------------
    st.subheader("📝 Test Your Own Review")

    user_input = st.text_area("Enter a product review:")

    if st.button("Predict"):
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("🚨 Fake Review Detected")
        else:
            st.success("✅ Genuine Review")
