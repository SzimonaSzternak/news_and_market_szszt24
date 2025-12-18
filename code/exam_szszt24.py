# Import libraries
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import ast
import tensorflow as tf

from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
from scipy.stats import linregress
from collections import Counter
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification


# Document-level Zipf function
def zipf_slope_single_article(df, row_id, text_col="text"):
    """
    Returns rank and frequency arrays for a single article.
    """
    text = df.loc[row_id, text_col].lower()
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    
    freq = np.array(sorted(word_counts.values(), reverse=True))
    rank = np.arange(1, len(freq) + 1)
    
    return rank, freq

def train_evaluate_lstm(X_train, y_train, X_test, y_test, max_words=5000, max_len=150, epochs=3, batch_size=32):
    """
    Train a simple LSTM classifier and evaluate on test set.
    Returns trained model and test accuracy.
    """
    # Tokenize
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    # Build LSTM model
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    model.fit(X_train_seq, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

    # Evaluate
    loss, acc = model.evaluate(X_test_seq, y_test)
    print(f"LSTM Test Accuracy: {acc:.3f}")

    return model, acc, tokenizer

def train_evaluate_bert(X_train, y_train, X_test, y_test, model_name="distilbert-base-uncased", epochs=1, batch_size=16):
    """
    Train a DistilBERT classifier and evaluate on test set.
    Returns trained model and test accuracy.
    """

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    test_encodings  = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train.values
    )).shuffle(1000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test.values
    )).batch(batch_size)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Evaluate
    loss, acc = model.evaluate(test_dataset)
    print(f"DistilBERT Test Accuracy: {acc:.3f}")

    return model, acc, tokenizer


if __name__ == "__main__":

    # Load in the datasets
    train = pd.read_csv("data/ModApte_train.csv")
    test  = pd.read_csv("data/ModApte_test.csv")

    # Convert topics from string to list
    train["topics"] = train["topics"].apply(ast.literal_eval)
    test["topics"]  = test["topics"].apply(ast.literal_eval)

    # Create binary label: 1 if 'earn' in topics, else 0
    train["label"] = train["topics"].apply(lambda x: 1 if "earn" in x else 0)
    test["label"]  = test["topics"].apply(lambda x: 1 if "earn" in x else 0)

    # X and y
    X_train, y_train = train["text"].fillna(""), train["label"]
    X_test, y_test   = test["text"].fillna(""), test["label"]

    # Use only the first 10 articles
    subset = train.iloc[:10].reset_index(drop=True)

    # Plot Zipf curves
    plt.figure(figsize=(8, 6))

    for i in range(10):
        rank, freq = zipf_slope_single_article(subset, i, text_col="text")
        plt.loglog(rank, freq, label=f"Article {i+1}")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf Curves for First 10 Reuters Articles")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # LSTM
    lstm_model, lstm_acc, lstm_tokenizer = train_evaluate_lstm(X_train, y_train, X_test, y_test, epochs=3)

    # DistilBERT
    bert_model, bert_acc, bert_tokenizer = train_evaluate_bert(X_train, y_train, X_test, y_test, epochs=1)

    # Store accuracy results
    results = pd.DataFrame(columns=["Model", "Accuracy"])

    # Add LSTM results
    results = results.append({"Model": "LSTM", "Accuracy": lstm_acc}, ignore_index=True)

    # Add BERT results
    results = results.append({"Model": "DistilBERT", "Accuracy": bert_acc}, ignore_index=True)

    results.to_csv("./results/model_comparison_results.csv", index=False)
