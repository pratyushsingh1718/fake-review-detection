import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from wordcloud import WordCloud

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Fake Review Detection", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Fake Product Review Detection System")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "SVM", "Random Forest"]
)

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("📂 Upload Dataset")
file = st.file_uploader("Upload CSV")

# -------------------------------
# DEFAULT PLACEHOLDERS (IMPORTANT)
# -------------------------------
model = None
vectorizer = None

if file:
    # -------------------------------
    # LOAD DATASET (NO HEADER FIX)
    # -------------------------------
    df = pd.read_csv(file, header=None, encoding='latin-1')
    df.columns = ["category", "rating", "label", "review_text"]

    df['label'] = df['label'].map({"OR": 0, "CG": 1})
    df.dropna(inplace=True)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # CLEAN TEXT
    # -------------------------------
    df['cleaned'] = df['review_text'].apply(clean_text)

    # -------------------------------
    # BALANCE DATASET
    # -------------------------------
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    if len(df_minority) > 0:
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        df = pd.concat([df_majority, df_minority_upsampled])

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    # -------------------------------
    # SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # MODEL
    # -------------------------------
    if model_choice == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000)

    elif model_choice == "SVM":
        model = SVC(probability=True, class_weight='balanced')

    else:
        model = RandomForestClassifier(class_weight='balanced')

    # -------------------------------
    # TABS UI
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model", "📝 Predict"])

    # -------------------------------
    # TAB 1 - DASHBOARD
    # -------------------------------
    with tab1:
        st.subheader("📊 Data Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Label Distribution")
            st.bar_chart(df['label'].value_counts())

        with col2:
            df['length'] = df['review_text'].apply(len)
            st.write("Review Length")
            st.bar_chart(df['length'])

        st.subheader("☁️ Word Cloud")
        text_data = " ".join(df['cleaned'])
        wordcloud = WordCloud(width=800, height=400).generate(text_data)
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)

    # -------------------------------
    # TAB 2 - MODEL
    # -------------------------------
    with tab2:
        st.subheader("🤖 Model Performance")

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2f}")
        c2.metric("Precision", f"{prec:.2f}")
        c3.metric("Recall", f"{rec:.2f}")
        c4.metric("F1 Score", f"{f1:.2f}")

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

# -------------------------------
# 🔥 MANUAL PREDICTOR (ALWAYS VISIBLE)
# -------------------------------
st.subheader("📝 Test Your Review")

user_input = st.text_area("Enter product review:")

if st.button("🔍 Analyze Review"):
    if model is None or vectorizer is None:
        st.warning("⚠️ Please upload dataset first")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        with st.spinner("Analyzing review..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            # Probability animation
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec)[0][1]
            else:
                prob = 0.5

            st.progress(prob)
            st.write(f"Fake Probability: {prob*100:.2f}%")

            if pred == 1:
                st.markdown("## 🚨 Fake Review Detected")
            else:
                st.markdown("## ✅ Genuine Review")
