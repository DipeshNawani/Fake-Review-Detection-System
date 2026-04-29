# =============================================================================
# sentiment_analysis.py
# SENTIMENT ANALYSIS MODULE — No external NLP library needed
# Uses Lexicon-Based approach + ML (Logistic Regression on TF-IDF)
# =============================================================================

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1 — LEXICON-BASED SENTIMENT
# No ML needed — works on any review instantly
# Based on positive/negative word dictionaries
# =============================================================================

POSITIVE_WORDS = {
    # Strong positive
    'excellent', 'outstanding', 'superb', 'exceptional', 'perfect', 'fantastic',
    'wonderful', 'brilliant', 'magnificent', 'splendid', 'flawless', 'phenomenal',
    # General positive
    'good', 'great', 'nice', 'best', 'awesome', 'amazing', 'love', 'loved',
    'beautiful', 'clean', 'comfortable', 'friendly', 'helpful', 'polite',
    'pleasant', 'enjoy', 'enjoyed', 'happy', 'satisfied', 'recommend',
    'recommended', 'impressed', 'impressive', 'delightful', 'lovely',
    'spacious', 'convenient', 'quick', 'fast', 'efficient', 'professional',
    'attentive', 'courteous', 'welcoming', 'cozy', 'charming', 'stunning',
    'delicious', 'tasty', 'fresh', 'affordable', 'value', 'worth',
    'relaxing', 'peaceful', 'quiet', 'calm', 'safe', 'secure',
    'modern', 'updated', 'renovated', 'spotless', 'immaculate',
}

NEGATIVE_WORDS = {
    # Strong negative
    'terrible', 'horrible', 'awful', 'dreadful', 'disgusting', 'atrocious',
    'appalling', 'abysmal', 'pathetic', 'unacceptable', 'catastrophic',
    # General negative
    'bad', 'worst', 'poor', 'dirty', 'nasty', 'rude', 'unfriendly',
    'unhelpful', 'slow', 'noisy', 'loud', 'uncomfortable', 'broken',
    'damaged', 'old', 'outdated', 'stained', 'smelly', 'smell', 'odor',
    'disappointing', 'disappointed', 'frustrating', 'frustrated', 'annoying',
    'annoyed', 'overpriced', 'expensive', 'waste', 'wasted', 'useless',
    'never', 'avoid', 'stay away', 'regret', 'regretted', 'mistake',
    'problem', 'issues', 'issue', 'complaint', 'complaints', 'unhappy',
    'dissatisfied', 'unpleasant', 'mediocre', 'subpar', 'lacking',
    'cold', 'small', 'tiny', 'cramped', 'stuffy', 'cockroach', 'bug', 'bugs',
    'mold', 'mould', 'leak', 'leaking', 'broken', 'not working', 'failed',
}

NEGATION_WORDS = {
    'not', 'no', "n't", 'never', 'neither', 'nor', 'nothing', 'nobody',
    'nowhere', 'hardly', 'barely', 'scarcely', 'without', 'lack', 'lacking',
}

INTENSIFIERS = {
    'very': 1.3, 'really': 1.3, 'extremely': 1.5, 'absolutely': 1.5,
    'totally': 1.4, 'completely': 1.4, 'highly': 1.3, 'super': 1.3,
    'so': 1.2, 'quite': 1.1, 'rather': 1.1, 'incredibly': 1.5,
    'exceptionally': 1.5, 'truly': 1.3, 'genuinely': 1.3,
    'somewhat': 0.8, 'slightly': 0.7, 'a bit': 0.8, 'little': 0.7,
}


def lexicon_sentiment(text):
    """
    Lexicon-based sentiment analysis.
    Returns: {'label': 'Positive'/'Negative'/'Neutral', 'score': float, 'pos_count': int, 'neg_count': int}

    How it works:
    1. Tokenize text into words
    2. Check each word against positive/negative dictionaries
    3. Apply intensifier multipliers (very, extremely, etc.)
    4. Apply negation flip (not good → negative)
    5. Score = pos_score - neg_score
    """
    if not isinstance(text, str) or text.strip() == '':
        return {'label': 'Neutral', 'score': 0.0, 'pos_count': 0, 'neg_count': 0}

    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b|n't", text_lower)

    pos_score  = 0.0
    neg_score  = 0.0
    pos_count  = 0
    neg_count  = 0

    i = 0
    while i < len(words):
        word = words[i]

        # Check intensifier at position i
        multiplier = INTENSIFIERS.get(word, 1.0)

        # Check negation in window of 3 words before current
        negated = any(words[max(0, i-3):i].count(neg) > 0 for neg in NEGATION_WORDS)

        # Look ahead for the actual sentiment word
        target = words[i+1] if multiplier != 1.0 and i+1 < len(words) else word
        target_idx = i+1 if multiplier != 1.0 and i+1 < len(words) else i

        if target in POSITIVE_WORDS:
            score = 1.0 * multiplier
            if negated:
                neg_score += score
                neg_count += 1
            else:
                pos_score += score
                pos_count += 1
            if target_idx != i:
                i += 1  # Skip next word since we used it

        elif target in NEGATIVE_WORDS:
            score = 1.0 * multiplier
            if negated:
                pos_score += score * 0.5  # "not bad" → slightly positive
                pos_count += 1
            else:
                neg_score += score
                neg_count += 1
            if target_idx != i:
                i += 1

        i += 1

    total = pos_score + neg_score
    if total == 0:
        compound = 0.0
    else:
        compound = (pos_score - neg_score) / (total + 0.5)  # Normalize to -1 to +1

    # Classify
    if compound >= 0.08:
        label = 'Positive'
    elif compound <= -0.08:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'label':     label,
        'score':     round(compound, 3),
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_score': round(pos_score, 2),
        'neg_score': round(neg_score, 2),
    }


def add_sentiment_column(df, review_col):
    """
    Add Sentiment column to dataframe using lexicon approach.
    Also adds Sentiment_Score column.
    """
    print("[SENTIMENT] Running lexicon-based sentiment analysis...")
    results = df[review_col].fillna('').apply(lexicon_sentiment)
    df = df.copy()
    df['Sentiment']       = results.apply(lambda x: x['label'])
    df['Sentiment_Score'] = results.apply(lambda x: x['score'])
    counts = df['Sentiment'].value_counts()
    print(f"[SENTIMENT] Positive: {counts.get('Positive',0)} | "
          f"Negative: {counts.get('Negative',0)} | "
          f"Neutral: {counts.get('Neutral',0)}")
    return df


# =============================================================================
# PART 2 — ML-BASED SENTIMENT CLASSIFIER
# Trains LR on TF-IDF features using lexicon labels as ground truth
# Better than pure lexicon for complex/sarcastic sentences
# =============================================================================

def train_sentiment_model(df, text_col='Clean_Review', sentiment_col='Sentiment'):
    """
    Train a Logistic Regression sentiment classifier.
    Uses Sentiment column (from lexicon) as training labels.

    Returns: trained pipeline, results dict
    """
    from sklearn.pipeline import Pipeline

    print("[SENTIMENT] Training ML sentiment classifier...")

    # Filter out neutral for binary training (optional — keep all 3)
    df_train = df[df[sentiment_col].isin(['Positive', 'Negative', 'Neutral'])].copy()

    X = df_train[text_col].fillna('')
    y = df_train[sentiment_col]

    if len(y.unique()) < 2:
        print("[SENTIMENT] Not enough classes to train. Skipping ML model.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc    = round(accuracy_score(y_test, y_pred) * 100, 2)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])

    print(f"[SENTIMENT] ML Sentiment Accuracy: {acc}%")

    results = {
        'accuracy': acc,
        'report':   report,
        'cm':       cm,
        'classes':  ['Positive', 'Neutral', 'Negative'],
        'y_test':   y_test,
        'y_pred':   y_pred,
    }

    return pipeline, results


def predict_sentiment(text, pipeline=None):
    """
    Predict sentiment of a single review.
    Uses ML pipeline if available, else falls back to lexicon.
    """
    from preprocessor import clean_text
    cleaned = clean_text(text)

    lexicon_result = lexicon_sentiment(text)

    if pipeline is not None and cleaned:
        try:
            ml_label = pipeline.predict([cleaned])[0]
            try:
                proba    = pipeline.predict_proba([cleaned])[0]
                classes  = pipeline.classes_
                conf     = round(max(proba) * 100, 1)
            except:
                conf = None
            return {
                'label':          ml_label,
                'confidence':     conf,
                'method':         'ML Classifier',
                'lexicon_label':  lexicon_result['label'],
                'lexicon_score':  lexicon_result['score'],
                'pos_score':      lexicon_result['pos_score'],
                'neg_score':      lexicon_result['neg_score'],
            }
        except Exception as e:
            pass

    # Fallback to lexicon
    return {
        'label':         lexicon_result['label'],
        'confidence':    round(abs(lexicon_result['score']) * 100, 1),
        'method':        'Lexicon-Based',
        'lexicon_score': lexicon_result['score'],
        'pos_score':     lexicon_result['pos_score'],
        'neg_score':     lexicon_result['neg_score'],
    }


# =============================================================================
# PART 3 — VISUALIZATIONS
# =============================================================================

SENTIMENT_COLORS = {
    'Positive': '#2ecc71',
    'Neutral':  '#3498db',
    'Negative': '#e74c3c',
}


def plot_sentiment_pie(df, sentiment_col='Sentiment'):
    """Sentiment distribution pie chart."""
    counts = df[sentiment_col].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    colors = [SENTIMENT_COLORS.get(l, '#95a5a6') for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        explode=[0.04]*len(labels), shadow=True,
        textprops={'fontsize': 11}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title('Sentiment Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_sentiment_by_rating(df, rating_col, sentiment_col='Sentiment'):
    """Sentiment breakdown per rating (1-5 stars)."""
    if rating_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ratings   = sorted(df[rating_col].dropna().unique())
    sentiments = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(ratings))
    width = 0.25

    for i, sent in enumerate(sentiments):
        counts = [
            len(df[(df[rating_col] == r) & (df[sentiment_col] == sent)])
            for r in ratings
        ]
        bars = ax.bar(x + i*width, counts, width,
                      label=sent, color=SENTIMENT_COLORS[sent],
                      edgecolor='white', linewidth=0.5)

    ax.set_title('Sentiment Distribution by Rating', fontsize=13, fontweight='bold')
    ax.set_xlabel('Rating (Stars)', fontsize=11)
    ax.set_ylabel('Number of Reviews', fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{int(r)} Star' for r in ratings])
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_sentiment_vs_fake(df, sentiment_col='Sentiment', fake_col='Fake_Label'):
    """How sentiment splits across Fake vs Genuine reviews."""
    if fake_col not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label_val, title) in zip(axes, [(0, 'Genuine Reviews'), (1, 'Fake Reviews')]):
        subset = df[df[fake_col] == label_val][sentiment_col].value_counts()
        colors = [SENTIMENT_COLORS.get(l, '#95a5a6') for l in subset.index]
        ax.pie(subset.values, labels=subset.index, colors=colors,
               autopct='%1.1f%%', startangle=140,
               textprops={'fontsize': 10})
        ax.set_title(f'Sentiment in {title}', fontsize=12, fontweight='bold')

    plt.suptitle('Sentiment Distribution: Fake vs Genuine', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sentiment_score_distribution(df, score_col='Sentiment_Score'):
    """Distribution of sentiment scores (continuous -1 to +1)."""
    if score_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}

    for sent, color in colors.items():
        subset = df[df['Sentiment'] == sent][score_col]
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=sent, edgecolor='white')

    ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.08,  color='#2ecc71', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.08, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title('Sentiment Score Distribution (-1 = Most Negative, +1 = Most Positive)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment Score', fontsize=11)
    ax.set_ylabel('Number of Reviews', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_top_sentiment_words(df, review_col='Clean_Review', sentiment_col='Sentiment', top_n=12):
    """Top words in Positive vs Negative reviews side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sent, color in zip(axes, ['Positive', 'Negative'], ['#2ecc71', '#e74c3c']):
        subset = df[df[sentiment_col] == sent][review_col].fillna('')
        all_words = ' '.join(subset).split()
        counts = Counter(all_words).most_common(top_n + 10)
        # Remove very generic words
        skip = {'product','item','hotel','room','place','thing','stay','good','bad'}
        filtered = [(w, c) for w, c in counts if w not in skip][:top_n]

        if filtered:
            words_list, count_list = zip(*filtered)
            ax.barh(list(reversed(words_list)), list(reversed(count_list)),
                    color=color, edgecolor='white', linewidth=0.4)
        ax.set_title(f'Top Words — {sent} Reviews', fontsize=12,
                     fontweight='bold', color=color)
        ax.set_xlabel('Frequency', fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.suptitle('Most Frequent Words by Sentiment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sentiment_cm(cm, classes):
    """Confusion matrix for ML sentiment classifier."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix — Sentiment Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)
    plt.tight_layout()
    return fig


# =============================================================================
# TEST
# =============================================================================
if __name__ == '__main__':
    test_reviews = [
        "The hotel was absolutely amazing! Staff were so friendly and helpful. Room was spotless.",
        "Terrible experience. Room was dirty and staff were extremely rude. Never coming back.",
        "It was okay. Nothing special but nothing bad either.",
        "The pool was great but the food was overpriced and bland.",
        "Not bad at all, surprisingly comfortable for the price.",
        "Very nice hotel very nice stay very nice rooms",
    ]
    print("LEXICON SENTIMENT TEST:")
    for r in test_reviews:
        res = lexicon_sentiment(r)
        print(f"  [{res['label']:8s}] score={res['score']:+.2f} | {r[:65]}")
        