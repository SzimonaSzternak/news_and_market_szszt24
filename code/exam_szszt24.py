import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay, 
    ConfusionMatrixDisplay
)
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.svm import LinearSVC

import nltk

# Download vader_lexicon to the default location
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer


# Load and label datasets
def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Load fake and true news datasets, assign labels, and merge them.
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1   # fake news
    true_df["label"] = 0   # real news

    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    return df

# Text cleaning function
def clean_text(text: str) -> str:
    """
    Clean text using regular expressions:
    - lowercase
    - remove URLs
    - remove numbers
    - remove punctuation
    - remove extra whitespace
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)                      # remove numbers
    text = re.sub(r"[^\w\s]", "", text)                  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()             # normalize spaces

    return text

# Extra text features
def add_extra_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Exclamation marks
    df["exclamation_count"] = df["content"].str.count("!")
    # Question marks
    df["question_count"] = df["content"].str.count("\?")
    # Uppercase ratio
    df["uppercase_ratio"] = df["content"].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x)+1e-6))
    return df

# Preprocess dataset
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined content field and clean text.
    """
    df = df.copy()

    # Combine title and text
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean content
    df["content_clean"] = df["content"].apply(clean_text)

    return df

# Balanced subsampling
def balanced_sample(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced dataset with equal fake and real samples.
    """
    fake = df[df["label"] == 1].sample(n=n_samples, random_state=random_state)
    real = df[df["label"] == 0].sample(n=n_samples, random_state=random_state)

    return pd.concat([fake, real]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def average_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip())>0]
    return np.mean([len(s.split()) for s in sentences]) if sentences else 0

def lexical_diversity(text):
    words = text.split()
    return len(set(words))/len(words) if len(words) > 0 else 0

# EDA: text length features
def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic text length features for EDA.
    """
    df = df.copy()
    df["word_count"] = df["content_clean"].apply(lambda x: len(x.split()))
    df["char_count"] = df["content_clean"].apply(len)
    return df

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx+1}: {', '.join(top_features)}")

# Sentiment analysis (VADER)
def compute_vader_sentiment(text: str, sia: SentimentIntensityAnalyzer) -> float:
    """
    Compute compound VADER sentiment score.
    """
    return sia.polarity_scores(text)["compound"]

def sentiment_bucket(compound_score):
    if compound_score <= -0.05:
        return "negative"
    elif compound_score >= 0.05:
        return "positive"
    else:
        return "neutral"


# Main execution
if __name__ == "__main__":

    # File paths
    FAKE_PATH = "data/Fake.csv"
    TRUE_PATH = "data/True.csv"

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Load data
    df = load_data(FAKE_PATH, TRUE_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    # Preprocess
    df = preprocess_data(df)
    print("Text preprocessing completed.")

    # Add extra text features (counts, uppercase ratio)
    df_balanced = add_extra_text_features(df)

    # Balanced sampling (adjust n_samples if needed)
    df_balanced = balanced_sample(df_balanced, n_samples=10000)
    print(f"Balanced dataset shape: {df_balanced.shape}")

    # Train-test split (for later modeling)
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced["content_clean"],
        df_balanced["label"],
        test_size=0.2,
        stratify=df_balanced["label"],
        random_state=42
    )
    # Convert label Series to NumPy arrays to allow boolean indexing on SciPy sparse matrices
    # (pandas Series doesn't implement .nonzero() which sparse indexing expects)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("Train-test split completed.")
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    # EDA: basic inspection
    print("\nClass distribution:")
    print(df_balanced["label"].value_counts())

    df_balanced = add_text_length_features(df_balanced)

    print("\nWord count summary by class:")
    print(df_balanced.groupby("label")["word_count"].describe())

    df_balanced["avg_sentence_length"] = df_balanced["content_clean"].apply(average_sentence_length)
    df_balanced["lexical_diversity"] = df_balanced["content_clean"].apply(lexical_diversity)

    # Ensure sentiment scores exist before bucketing (compute if missing)
    if "sentiment" not in df_balanced.columns:
        sia = SentimentIntensityAnalyzer()
        df_balanced["sentiment"] = df_balanced["content_clean"].apply(
            lambda x: compute_vader_sentiment(x, sia)
        )

    df_balanced["sentiment_bucket"] = df_balanced["sentiment"].apply(sentiment_bucket)

    print(df_balanced.groupby(["label", "sentiment_bucket"]).size().unstack(fill_value=0))

    df_balanced.groupby(["label", "sentiment_bucket"]).size().unstack().plot(kind="bar", figsize=(6,4))
    plt.title("Sentiment Polarity Distribution by Class")
    plt.xlabel("Label (0=Real, 1=Fake)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/sentiment_bucket_distribution.png", dpi=300)
    plt.close()


    # EDA: word count distribution
    plt.figure()
    for label, name in zip([0, 1], ["Real", "Fake"]):
        subset = df_balanced[df_balanced["label"] == label]
        plt.hist(
            subset["word_count"],
            bins=50,
            alpha=0.6,
            label=name
        )

    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.title("Word Count Distribution: Real vs Fake News")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/word_count_distribution.png", dpi=300)
    plt.close()

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()

    df_balanced["sentiment"] = df_balanced["content_clean"].apply(
        lambda x: compute_vader_sentiment(x, sia)
    )

    print("\nSentiment summary by class:")
    print(df_balanced.groupby("label")["sentiment"].describe())

    # Sentiment distribution plot
    plt.figure()
    for label, name in zip([0, 1], ["Real", "Fake"]):
        subset = df_balanced[df_balanced["label"] == label]
        plt.hist(
            subset["sentiment"],
            bins=50,
            alpha=0.6,
            label=name
        )

    plt.xlabel("VADER Sentiment (compound)")
    plt.ylabel("Frequency")
    plt.title("Sentiment Distribution: Real vs Fake News")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/sentiment_distribution.png", dpi=300)
    plt.close()

    # Save processed dataset with sentiment scores
    df_balanced.to_csv("results/processed_fake_real_news.csv", index=False)
    print("Processed dataset saved to 'results/processed_fake_real_news.csv'")

    # TF-IDF + Logistic Regression
    # 1. TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,  # limit to top 10k terms
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 2. Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # 3. Predictions
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

    # 4. Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\nClassification Results (Logistic Regression + TF-IDF):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Top 20 words for fake vs real
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pos_idx = coefs.argsort()[-20:][::-1]  # most positive (fake)
    top_neg_idx = coefs.argsort()[:20]         # most negative (real)

    print("\nTop 20 words indicative of Fake News:")
    print(feature_names[top_pos_idx])

    print("\nTop 20 words indicative of Real News:")
    print(feature_names[top_neg_idx])

    # Save top words indicative of Fake vs Real news
    top_words_df = pd.DataFrame({
        "Fake_news_top_words": feature_names[top_pos_idx],
        "Real_news_top_words": feature_names[top_neg_idx]
    })

    top_words_df.to_csv("results/top_words_fake_real.csv", index=False)
    print("\nTop words saved to 'top_words_fake_real.csv'")

    fake_tfidf = X_train_tfidf[y_train==1]
    real_tfidf = X_train_tfidf[y_train==0]

    lda_fake = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_fake.fit(fake_tfidf)
    print("\nTop words per topic (Fake News):")
    # collect topics for saving
    feature_names_arr = vectorizer.get_feature_names_out()
    lda_fake_topics = []
    for topic_idx, topic in enumerate(lda_fake.components_):
        top_features = [feature_names_arr[i] for i in topic.argsort()[:-10 - 1:-1]]
        lda_fake_topics.append(top_features)
        print(f"Topic {topic_idx+1}: {', '.join(top_features)}")
    # save fake LDA topics
    with open(os.path.join("results", "lda_fake_topics.txt"), "w", encoding="utf-8") as f:
        for i, toks in enumerate(lda_fake_topics, 1):
            f.write(f"Topic {i}: {', '.join(toks)}\n")

    lda_real = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_real.fit(real_tfidf)
    print("\nTop words per topic (Real News):")
    lda_real_topics = []
    for topic_idx, topic in enumerate(lda_real.components_):
        top_features = [feature_names_arr[i] for i in topic.argsort()[:-10 - 1:-1]]
        lda_real_topics.append(top_features)
        print(f"Topic {topic_idx+1}: {', '.join(top_features)}")
    # save real LDA topics
    with open(os.path.join("results", "lda_real_topics.txt"), "w", encoding="utf-8") as f:
        for i, toks in enumerate(lda_real_topics, 1):
            f.write(f"Topic {i}: {', '.join(toks)}\n")


    # Confusion matrix plot
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5,5))
    cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix: Logistic Regression")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to 'results/confusion_matrix.png'")

    # ROC curve plot
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve: Logistic Regression")
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=300)
    plt.close()
    print("ROC curve saved to 'results/roc_curve.png'")

    pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = cross_val_score(pipeline, df_balanced["content_clean"], df_balanced["label"], cv=cv, scoring='f1')
    roc_scores = cross_val_score(pipeline, df_balanced["content_clean"], df_balanced["label"], cv=cv, scoring='roc_auc')

    print(f"\n5-fold CV F1: {f1_scores.mean():.4f}")
    print(f"5-fold CV ROC-AUC: {roc_scores.mean():.4f}")
    # Save cross-validation scores
    cv_out = {
        "f1_scores": f1_scores.tolist(),
        "f1_mean": float(f1_scores.mean()),
        "roc_scores": roc_scores.tolist(),
        "roc_mean": float(roc_scores.mean())
    }
    with open(os.path.join("results", "cv_scores.json"), "w", encoding="utf-8") as f:
        json.dump(cv_out, f, indent=2)

    # Wrap your trained classifier
    # Some sklearn versions don't accept the string 'prefit' for cv; use a small int for cross-validation instead
    calibrated_clf = CalibratedClassifierCV(clf, cv=3)
    calibrated_clf.fit(X_train_tfidf, y_train)

    # Evaluate calibration
    disp = CalibrationDisplay.from_estimator(calibrated_clf, X_test_tfidf, y_test, n_bins=10)
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig("results/calibration_curve.png", dpi=300)
    plt.close()
    print("Calibration curve saved to 'results/calibration_curve.png'")

    svc = LinearSVC(max_iter=5000, random_state=42)
    svc.fit(X_train_tfidf, y_train)
    y_pred_svc = svc.predict(X_test_tfidf)

    # Print metrics in a small table
    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "LinearSVC"],
        "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_svc)],
        "Precision": [precision_score(y_test, y_pred), precision_score(y_test, y_pred_svc)],
        "Recall": [recall_score(y_test, y_pred), recall_score(y_test, y_pred_svc)],
        "F1": [f1_score(y_test, y_pred), f1_score(y_test, y_pred_svc)]
    })

    print("\nComparison of classifiers:")
    print(metrics_df)
    # Save comparison metrics and models
    metrics_df.to_csv(os.path.join("results", "metrics_comparison.csv"), index=False)

    # Save vectorizer and models for reproducibility
    try:
        joblib.dump(vectorizer, os.path.join("results", "tfidf_vectorizer.joblib"))
        joblib.dump(clf, os.path.join("results", "logistic_clf.joblib"))
        joblib.dump(svc, os.path.join("results", "linear_svc.joblib"))
    except Exception as e:
        print(f"Warning: failed to save models/vectorizer: {e}")

    # Save LDA topics and top words (top_words_df already saved earlier)
    # top_words_df saved to results/top_words_fake_real.csv earlier
    print("Saved metrics, models and artifacts to the 'results' folder.")