# Fake vs Real News Analysis  
**Course:** DSK 822 – News and Market Sentiment Analytics  
**Author:** Szimona Szternak (szszt24)  
**University:** University of Southern Denmark  

---

## Project Overview
This project analyzes stylistic, sentiment, and thematic differences between fake and real news articles. In addition to exploratory data analysis and sentiment modeling, multiple machine learning classifiers are trained to distinguish between fake and real news using TF-IDF features. The project also discusses how sentiment-driven trading or decision models may be biased by systematic differences in fake news content.

---

## Dataset
The analysis uses the **Fake and Real News Dataset** from Kaggle:

- **Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data  
- **Files:**
  - `Fake.csv` – Fake news articles  
  - `True.csv` – Real news articles  
- **Columns:** `title`, `text`, `subject`, `date`  
- **Total samples:** 44,898 (before balancing)

---

## Methods and Workflow

### Data Preprocessing
Article titles and body texts were combined into a single content field and cleaned by lowercasing, removing URLs, numbers, punctuation, and extra whitespace. Prior to text cleaning, several stylistic features were extracted, including punctuation counts, uppercase ratio, average sentence length, and lexical diversity. To ensure comparability between classes, the dataset was balanced by randomly sampling 10,000 fake and 10,000 real news articles.

### Exploratory Data Analysis (EDA)
Exploratory analysis examined class balance, word count distributions, and sentiment differences between fake and real news. Fake news articles were found to be slightly longer on average and to exhibit more negative sentiment. Stylistic features such as punctuation usage and capitalization also differed systematically between the two classes.

### Sentiment Analysis
Sentiment was computed using the VADER sentiment analyzer. Compound scores were stored and articles were categorized into negative, neutral, and positive sentiment buckets. The distribution of sentiment polarity differed across classes, with fake news showing a higher proportion of negative sentiment. These findings highlight how sentiment-driven trading or decision models may be biased if fake news is not explicitly accounted for.

### Feature Engineering
Textual features were generated using TF-IDF vectorization with the top 10,000 terms, including unigrams and bigrams, and removing stop words. In addition to TF-IDF features, numeric stylistic features such as punctuation counts, uppercase ratio, average sentence length, and lexical diversity were retained for descriptive analysis.

### Topic Modeling
Topic modeling was performed using Latent Dirichlet Allocation (LDA) with five topics per class. Two complementary approaches were used. First, LDA was fitted separately on fake and real news TF-IDF matrices to compare high-level thematic structure. Second, a more detailed topic inspection was conducted by extracting and saving the top words for each topic per class. Fake news topics were dominated by political figures, sensational phrasing, and conspiracy-related terms, while real news topics focused on institutional actors, geopolitical issues, and reporting language. These results demonstrate clear thematic separation between fake and real news content.

### Classification
The balanced dataset of 20,000 articles was split into training (16,000 samples) and test (4,000 samples) sets. A Logistic Regression classifier trained on TF-IDF features achieved very high performance, with an F1-score above 0.98 and a ROC-AUC close to 1.0. The most indicative words for fake news included emotionally charged and sensational terms, while real news was characterized by formal reporting language and newswire references.

To benchmark performance, a LinearSVC classifier was trained using the same TF-IDF representation. LinearSVC slightly outperformed Logistic Regression across all evaluation metrics, confirming the robustness of linear models for this task.

### Cross-Validation and Calibration
Model stability was assessed using 5-fold stratified cross-validation, yielding consistently high F1 and ROC-AUC scores. Probability calibration was performed using `CalibratedClassifierCV`, and the resulting calibration curve showed good alignment between predicted probabilities and observed outcomes.

---

## Repository Structure
```text
.
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── code/
│   └── exam_szszt24.py
│
├── results/
│   ├── processed_fake_real_news.csv
│   ├── top_words_fake_real.csv
│   ├── lda_fake_topics.txt
│   ├── lda_real_topics.txt
│   ├── metrics_comparison.csv
│   ├── cv_scores.json
│   ├── word_count_distribution.png
│   ├── sentiment_distribution.png
│   ├── sentiment_bucket_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── calibration_curve.png
│
├── report/
│   ├── report_SzimonaSzternak_szszt24.pdf
│
└── README.md