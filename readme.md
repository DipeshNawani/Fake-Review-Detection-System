# Fake Review Detection System
### Machine Learning + Deep Learning + Sentiment Analysis

> An end-to-end automated system to detect fake online reviews using rule-based heuristics, traditional Machine Learning, Bidirectional LSTM Deep Learning, and Sentiment Analysis, deployed as an interactive Streamlit dashboard.

## Live Demo

streamlit run app.py
Open `http://localhost:8501` in your browser.

## What This Project Does

Upload **any** product, hotel, app, or food review CSV dataset and the system will:

- **Auto-detect** which column contains the review text (no manual config needed)
- **Generate fake labels** automatically using 8 rule-based heuristics - no pre-labeled data required
- **Train 3 ML models** - Logistic Regression, Naive Bayes, SVM using TF-IDF features
- **Train a Bidirectional LSTM** deep learning model for sequential text understanding
- **Perform Sentiment Analysis** - Positive / Neutral / Negative using lexicon + ML
- **Display everything** in a 6-tab interactive dashboard with charts, confusion matrices, and live prediction

## Project Structure

fake-review-detection/
│
├── app.py                  ← Main Streamlit dashboard (run this)
├── preprocessor.py         ← Text cleaning + auto column detection + 8-rule fake label generation
├── ml_model.py             ← Logistic Regression, Naive Bayes, SVM + TF-IDF
├── dl_model.py             ← Bidirectional LSTM (TensorFlow/Keras)
├── sentiment_analysis.py   ← Lexicon-based + ML sentiment classifier
├── visualizations.py       ← All charts and graphs
├── requirements.txt        ← Python dependencies
└── README.md

## Supported Datasets

The system works with **any** review CSV - no column renaming needed:

| Dataset                           | Source                        |
|-----------------------------------|-------------------------------|
| Flipkart Product Reviews          |           Kaggle              |
| Datafiniti Hotel Reviews          |           Kaggle              |
| Google Play Store Reviews         |           Kaggle              |
| Zomato / Food Reviews             |           Kaggle              |
| IMDb Movie Reviews                |           Kaggle              |
| Any CSV with a review/text column |             -                 |
|-----------------------------------|-------------------------------|

## Installation

### 1. Clone the repository
git clone https://github.com/DipeshNawani/Fake-Review-Detection-System.git
cd Fake-Review-Detection-System

### 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py

## How It Works

### Step 1 - Auto Column Detection
Three-pass algorithm identifies the review column automatically:
- Pass 1: Exact name match (`reviews.text`, `review`, `translated_review`)
- Pass 2: Substring match excluding metadata columns (date, url, id, city)
- Pass 3: Fallback selects the text column with longest average length

### Step 2 - Text Preprocessing
Raw Text → Lowercase → Remove URLs → Remove symbols/numbers → Remove stopwords → Clean_Review column

### Step 3 - Fake Label Generation (8 Rules)

|----------------------|-------------------------------------------|--------------------------------------------------------------|
| Rule                 | Condition                                 | Example                                                      |
|----------------------|-------------------------------------------|--------------------------------------------------------------|
| Too Short            | Review < 15 characters                    | "ok", "nice"                                                 |
| Repeated Words       | 40%+ meaningful words identical           | "clean clean clean clean service clean"                      |
| Duplicate Phrase     | Same bigram 3+ times or half-text repeats | "very nice hotel very nice stay very nice rooms"             |
| Rating Mismatch      | 5-star + Negative or 1-star + Positive    | Rating=5, Review="Worst stay ever"                           |
| Generic Only         | Only vague words, no specific nouns       | "good product nice item great stuff"                         |
| Excessive Caps       | 3+ "!" or 60% words ALL_CAPS              | "AMAZING!!! BUY NOW!!!"                                      |
| Exaggerated Language | "1000%", "best ever", "must buy"          | "1000% recommend must visit"                                 |
| No Specific Details  | 15+ words but zero concrete nouns         | "Very amazing wonderful fantastic experience overall great"  |
|----------------------|-------------------------------------------|--------------------------------------------------------------|

**Any one rule triggered -> FAKE (1). All rules clear -> GENUINE (0)**

### Step 4 - ML Models (TF-IDF Features)
- **Logistic Regression** - Linear decision boundary in TF-IDF space
- **Naive Bayes** - Word probability-based classification
- **SVM (LinearSVC)** - Maximum-margin hyperplane separator

### Step 5 - Deep Learning (Bidirectional LSTM)
Text → Tokenize → Pad to length 100 → Embedding (64-dim)
     → BiLSTM (64 units, forward + backward) → Dropout (0.3)
     → Dense (32, ReLU) → Dropout (0.3) → Dense (1, Sigmoid)
     → Output > 0.5 = FAKE  |  Output ≤ 0.5 = GENUINE

### Step 6 - Sentiment Analysis
- **Lexicon-based**: Custom dictionaries of 100+ positive/negative words with intensifier multipliers and negation handling
- **ML Classifier**: Logistic Regression trained on lexicon-generated labels → **88% accuracy**

## Results

### Fake Detection (Datafiniti Hotel Reviews, n=5,000)

|----------------------|----------|-----------|--------|----------|
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 72.4%    | 73.92%    | 72.4%  | 72.93%   |
| Naive Bayes          | 71.9%    | 69.93%    | 71.9%  | 68.25%   |
| SVM (LinearSVC)      | 70.6%    | 71.18%    | 70.6%  | 70.85%   |
| Bidirectional LSTM   | ~88-93%  | -         | -      | -        |
|----------------------|----------|-----------|--------|----------|

### Sentiment Analysis

|-------------------------|----------|
| Method                  | Accuracy |
|-------------------------|----------|
| Lexicon-Based           | ~80%     |
| ML Sentiment Classifier | 88.0%    |
|-------------------------|----------|

## Dashboard - 6 Tabs

|----------------------|--------------------------------------------------------------------------|
| Tab                  | Contents                                                                 |
|----------------------|--------------------------------------------------------------------------|
| Dataset Overview     | KPI cards, column detection results, processed data sample, downloads    |
| EDA and Visuals      | Pie chart, rating dist, word frequency, review length, rule triggers     |
| Sentiment Analysis   | Sentiment pie, score dist, by-rating breakdown, live predictor           |
| ML Results           | Summary table, accuracy comparison, per-model confusion matrix           |
| Deep Learning        | Architecture, training curves, confusion matrix, ML vs DL comparison     |
| Live Prediction      | Rule-based + ML + DL prediction, confidence %, batch CSV prediction      |
|----------------------|--------------------------------------------------------------------------|

## Tech Stack

|--------------------------|------------------------------------------|
| Technology               | Purpose                                  |
|--------------------------|------------------------------------------|
| Python 3.10+             | Core language                            |
| Streamlit 1.20+          | Interactive web dashboard                |
| Scikit-learn 1.2+        | ML models, TF-IDF, evaluation metrics    |
| TensorFlow / Keras 2.10+ | Bidirectional LSTM                       |
| Pandas / NumPy           | Data manipulation                        |
| Matplotlib / Seaborn     | Visualizations                           |
|--------------------------|------------------------------------------|

## Dataset
**This project does not require any specific dataset.** *It works with any CSV file that contains a review or text column.*
The datasets listed above (Flipkart, Hotel, PlayStore, Zomato, IMDb) are just examples, The system automatically detects the review column, generates fake labels, and trains all models regardless of the dataset structure or domain.

For testing and development, the following dataset was used:
Datafiniti Hotel Reviews - available on Kaggle: (https://www.kaggle.com/datasets/datafiniti/hotel-reviews)
***The dataset file is not included in this repository due to size constraints. Download from Kaggle and upload directly through the dashboard interface.***

## Key Highlights

- No pre-labeled data needed - 8-rule automatic labeling
- Universal column detection - works across any review dataset
- Dual analysis - fake detection AND sentiment in one pipeline
- Explainable predictions - shows exactly which rule flagged a review
- Batch prediction - predict all rows of a new CSV at once

## Author
**Dipesh Nawani**
dipeshnawani160@gmail.com
