# Import libraries
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Use tf-keras for TensorFlow Hub compatibility

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import ast
import tensorflow as tf
import torch
import tensorflow_hub as hub
import tensorflow_text as text  # Required for BERT preprocessing

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
from scipy.stats import linregress
from collections import Counter
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration

# Create results folder if it does not exist
os.makedirs("./results", exist_ok=True)

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

def train_evaluate_bert(X_train, y_train, X_test, y_test, epochs=1, batch_size=16):
    """
    Train a BERT classifier using TensorFlow Hub and evaluate on test set.
    Returns trained model and test accuracy.
    """

    # TensorFlow Hub BERT models (small_bert is faster for training)
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"

    # Build model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        X_train.values,
        y_train.values
    )).shuffle(1000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        X_test.values,
        y_test.values
    )).batch(batch_size)

    # Train
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Evaluate
    loss, acc = model.evaluate(test_dataset)
    print(f"BERT Test Accuracy: {acc:.3f}")

    return model, acc, None  # No separate tokenizer needed

def zero_shot_classifier(texts, candidate_labels, model_name='all-MiniLM-L6-v2'):
    """
    Zero-shot classifier using sentence embeddings.

    Args:
        texts (list or pd.Series): List of texts to classify.
        candidate_labels (list): List of candidate labels/topics.
        model_name (str): HuggingFace sentence-transformers model.

    Returns:
        predictions (list): Predicted labels for each text.
        similarity_scores (list of lists): Cosine similarity scores for each label.
    """
    # Load sentence-transformer model
    model = SentenceTransformer(model_name)

    # Encode texts and candidate labels
    text_embeddings = model.encode(list(texts), convert_to_tensor=True)
    label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)

    # Compute cosine similarity
    cos_sim = util.cos_sim(text_embeddings, label_embeddings)  # shape: [num_texts, num_labels]

    # For each text, pick the label with highest similarity
    predictions = []
    similarity_scores = cos_sim.cpu().numpy().tolist()  # optional: save all scores

    for sim in cos_sim:
        best_idx = sim.argmax()
        predictions.append(candidate_labels[best_idx])

    return predictions, similarity_scores

def compute_sentiment_scores(texts):
    """
    Compute sentiment scores for a list or pd.Series of texts.
    
    Args:
        texts (list or pd.Series): Texts to score.
    
    Returns:
        List of compound sentiment scores (-1 to +1)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts]
    return scores

def rag_generate(query, corpus_texts, embed_model, generator_model, generator_tokenizer, top_k=3):
    """
    Retrieves top_k articles relevant to query and generates a response.
    """
    # Encode query
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    corpus_embeddings = embed_model.encode(corpus_texts, convert_to_tensor=True)

    # Compute cosine similarity and retrieve top-k
    cos_sim = util.cos_sim(query_embedding, corpus_embeddings)
    top_k_idx = torch.topk(cos_sim, k=top_k).indices[0]
    retrieved_texts = [corpus_texts[i] for i in top_k_idx]

    # Concatenate retrieved texts as context
    context = " ".join(retrieved_texts)
    input_text = f"question: {query} context: {context}"

    # Tokenize and generate
    inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = generator_model.generate(**inputs, max_length=150, num_beams=3)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

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

    # Zipf slope for first 10 articles
    # Use only the first 10 articles
    subset = train.iloc[:10].reset_index(drop=True)

    # Plot Zipf curves and save plot to results folder
    plt.figure(figsize=(8, 6))

    for i in range(10):
        rank, freq = zipf_slope_single_article(subset, i, text_col="text")
        plt.loglog(rank, freq, label=f"Article {i+1}")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf Curves for First 10 Reuters Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./results/zipf_curves_first_10_articles.png", dpi=300)

    # LSTM vs BERT
    # LSTM
    lstm_model, lstm_acc, lstm_tokenizer = train_evaluate_lstm(X_train, y_train, X_test, y_test, epochs=3)

    # DistilBERT
    bert_model, bert_acc, bert_tokenizer = train_evaluate_bert(X_train, y_train, X_test, y_test, epochs=1)

    # Print for verification
    print("LSTM Accuracy:", lstm_acc)
    print("BERT Accuracy:", bert_acc)

    # Zero-shot classifier
    # Candidate labels (topics you want to classify)
    candidate_labels = ["earn", "acq", "crude", "trade", "interest"]

    # Apply zero-shot classifier
    texts = train['text'].fillna("")[:100]  # you can use full dataset later
    preds, scores = zero_shot_classifier(texts, candidate_labels)

    # Check predictions
    for t, p in zip(texts[:5], preds[:5]):
        print(f"Text: {t[:50]}... -> Predicted: {p}")

    y_true = train['label'][:len(preds)]  # 1 if 'earn' else 0
    y_pred = [1 if p=='earn' else 0 for p in preds]

    zero_shot_acc = accuracy_score(y_true, y_pred)
    zero_shot_f1  = f1_score(y_true, y_pred)

    # Print for verification
    print("Zero-Shot Accuracy:", zero_shot_acc)
    print("Zero-Shot F1-score:", zero_shot_f1)

    # Store accuracy results
    results = pd.DataFrame([
        {"Model": "LSTM", "Accuracy": lstm_acc},
        {"Model": "BERT", "Accuracy": bert_acc},
        {"Model": "Zero-Shot Embeddings", "Accuracy": zero_shot_acc}
    ])
    
    # Save model comparison results to results folder
    results.to_csv("./results/model_comparison_results.csv", index=False)
    print("Model comparison results saved to ./results/model_comparison_results.csv")

    train['sentiment'] = compute_sentiment_scores(train['text'].fillna(""))

    # Select columns and save to CSV
    output_df = train[['title', 'text', 'sentiment']]
    output_df.to_csv("./results/train_sentiment_scores.csv", index=False)
    print("Sentiment scores saved to ./results/train_sentiment_scores.csv")

    test['sentiment'] = compute_sentiment_scores(test['text'].fillna(""))
    output_test_df = test[['title', 'text', 'sentiment']]
    output_test_df.to_csv("./results/test_sentiment_scores.csv", index=False)
    print("Sentiment scores saved to ./results/test_sentiment_scores.csv")

    # RAG retrieval + generation
    print("\n--- RAG Example ---")
    # Load embedding model and generator (BART)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    generator_model     = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    sample_query = "What articles mention earnings?"
    rag_answer = rag_generate(sample_query, train['text'].fillna("").tolist(), embed_model, generator_model, generator_tokenizer, top_k=3)
    print("Query:", sample_query)
    print("RAG Answer:", rag_answer)