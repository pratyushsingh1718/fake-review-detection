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

    df = pd.read_csv(file, header=None, encoding='latin-1')
    df.columns = ["category", "rating", "label", "review_text"]

    df['label'] = df['label'].map({"OR": 0, "CG": 1})
    df.dropna(inplace=True)

    df['cleaned'] = df['review_text'].apply(clean_text)
    df['length'] = df['review_text'].apply(len)

    df = df[df['length'] >= min_length]

    Q1 = df['length'].quantile(0.25)
    Q3 = df['length'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['length'] >= Q1 - 1.5*IQR) & (df['length'] <= Q3 + 1.5*IQR)]

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

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    text_features = vectorizer.fit_transform(df['cleaned'])

    extra_features = df[['word_count', 'exclamation_count', 'caps_ratio', 'exaggeration']].values

    from scipy.sparse import hstack
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

    with tab1:

        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']

        fig = px.bar(label_counts, x='Label', y='Count', color='Label', text='Count',
                     title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x="length", nbins=50, title="Review Length Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("☁️ Word Cloud")
        text_data = " ".join(df['cleaned'])
        wordcloud = WordCloud(width=800, height=400).generate(text_data)
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)

        corr = df[['rating', 'length', 'label']].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:

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

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

    with tab3:

        st.subheader("📝 Test Your Review")

        user_input = st.text_area("Enter product review:")

        if st.button("🔍 Analyze Review"):

            if user_input.strip() == "":
                st.warning("⚠️ Please enter a review")

            else:
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])

                pred = model.predict(vec)[0]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(vec)[0][1]
                else:
                    prob = 0.5

                st.progress(prob)
                st.write(f"Fake Probability: {prob*100:.2f}%")

                if pred == 1:
                    st.error("🚨 Fake Review Detected")
                else:
                    st.success("✅ Genuine Review")

else:
    st.info("⬆️ Please upload a dataset to begin.")
