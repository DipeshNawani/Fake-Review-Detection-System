# =============================================================================
# ml_model.py
# MACHINE LEARNING PART — Logistic Regression, Naive Bayes, SVM
# TF-IDF Vectorization + Training + Evaluation
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI ke bina bhi kaam kare
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: DATA PREPARATION
# Clean_Review aur Fake_Label ko train/test split karo
# =============================================================================

def prepare_data(df, text_col='Clean_Review', label_col='Fake_Label', test_size=0.2):
    """
    Dataset ko train aur test mein split karo.
    80% training, 20% testing.
    """
    print("[ML] Preparing data for training...")

    X = df[text_col].fillna('')
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y  # Class balance maintain karo
    )

    print(f"[ML] Training samples : {len(X_train)}")
    print(f"[ML] Testing  samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 2: TF-IDF VECTORIZER
# Text ko numbers mein convert karo — ML ke liye zaroori hai
# =============================================================================

def build_tfidf_vectorizer(max_features=5000):
    """
    TF-IDF Vectorizer banao.
    min_df=1 so it works on any dataset size without 'no terms remain' error.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,        # Accept every word — prevents 'no terms remain' error
        max_df=0.95,
        sublinear_tf=True
    )
    return vectorizer


# =============================================================================
# STEP 3: MODEL DEFINITIONS
# Teeno ML models define karo
# =============================================================================

def get_all_models():
    """
    Teeno ML models return karo as a dict.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            class_weight='balanced'  # Imbalanced dataset handle karo
        ),
        'Naive Bayes': MultinomialNB(
            alpha=0.1  # Smoothing parameter
        ),
        'SVM': LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )
    }
    return models


# =============================================================================
# STEP 4: TRAIN + EVALUATE ALL MODELS
# =============================================================================

def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Teeno models ko train karo aur evaluate karo.
    
    Returns:
        results_dict: {model_name: {accuracy, precision, recall, f1, cm, report}}
        pipelines: {model_name: trained_pipeline}  ← Live prediction ke liye
    """
    print("\n" + "=" * 50)
    print("ML MODEL TRAINING STARTED")
    print("=" * 50)

    vectorizer = build_tfidf_vectorizer()
    models = get_all_models()

    results = {}
    pipelines = {}

    for model_name, model in models.items():
        print(f"\n[ML] Training: {model_name}...")

        # Pipeline banao: TF-IDF → Model
        # Pipeline ka fayda: ek call mein vectorize + predict hota hai
        pipeline = Pipeline([
            ('tfidf', build_tfidf_vectorizer()),
            ('classifier', model)
        ])

        # Train karo
        pipeline.fit(X_train, y_train)

        # Predict karo
        y_pred = pipeline.predict(X_test)

        # Metrics calculate karo
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)
        report    = classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'], zero_division=0)

        results[model_name] = {
            'accuracy':  round(acc * 100, 2),
            'precision': round(precision * 100, 2),
            'recall':    round(recall * 100, 2),
            'f1_score':  round(f1 * 100, 2),
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
        pipelines[model_name] = pipeline

        print(f"  ✅ Accuracy  : {acc*100:.2f}%")
        print(f"     Precision : {precision*100:.2f}%")
        print(f"     Recall    : {recall*100:.2f}%")
        print(f"     F1-Score  : {f1*100:.2f}%")

    print("\n" + "=" * 50)
    print("ML MODEL TRAINING COMPLETE")
    print("=" * 50)

    return results, pipelines


# =============================================================================
# STEP 5: VISUALIZATION FUNCTIONS
# Graphs jo examiner ko dikhane hain
# =============================================================================

def plot_confusion_matrix(cm, model_name):
    """
    Confusion matrix plot karo.
    Returns: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Genuine', 'Fake'],
        yticklabels=['Genuine', 'Fake'],
        ax=ax,
        linewidths=0.5
    )

    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()

    return fig


def plot_ml_accuracy_comparison(results):
    """
    Teeno ML models ki accuracy comparison bar chart.
    Returns: matplotlib figure object
    """
    model_names = list(results.keys())
    accuracies  = [results[m]['accuracy'] for m in model_names]
    f1_scores   = [results[m]['f1_score'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)',
                   color='#4C72B0', edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)',
                   color='#DD8452', edgecolor='white', linewidth=0.7)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('ML Models — Accuracy & F1-Score Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    return fig


def plot_metrics_table(results):
    """
    Teeno models ke metrics ek table mein dikhao.
    Returns: pandas DataFrame (Streamlit mein directly show kar sakte ho)
    """
    rows = []
    for model_name, res in results.items():
        rows.append({
            'Model': model_name,
            'Accuracy (%)': res['accuracy'],
            'Precision (%)': res['precision'],
            'Recall (%)': res['recall'],
            'F1-Score (%)': res['f1_score']
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 6: LIVE PREDICTION USING TRAINED ML MODEL
# User koi bhi review type kare → Fake ya Genuine batao
# =============================================================================

def predict_single_review_ml(review_text, pipeline, model_name):
    """
    Single review ke liye prediction karo.
    
    Input : Review text (string)
    Output: {label, confidence, model}
    """
    from preprocessor import clean_text

    # Clean karo
    cleaned = clean_text(review_text)

    if not cleaned:
        return {
            'label': 'Unable to predict',
            'confidence': 0,
            'model': model_name,
            'cleaned_review': cleaned
        }

    # Predict karo
    prediction = pipeline.predict([cleaned])[0]

    # Confidence (probability) — SVM ke liye direct probability nahi hoti
    try:
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = round(max(proba) * 100, 1)
    except:
        confidence = None  # SVM ka case

    label = 'FAKE' if prediction == 1 else 'GENUINE'

    return {
        'label': label,
        'prediction_raw': int(prediction),
        'confidence': confidence,
        'model': model_name,
        'cleaned_review': cleaned
    }


# =============================================================================
# TEST — Directly "python ml_model.py" run karne pe chalega
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from preprocessor import full_preprocessing_pipeline

    # Dataset load karo
    df_raw = pd.read_csv('../fake_review_dataset_with_labels.csv')
    df_clean, col_info = full_preprocessing_pipeline(df_raw)

    # Train/test split
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # Train all models
    results, pipelines = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # Metrics table print karo
    print("\nFINAL RESULTS TABLE:")
    print(plot_metrics_table(results).to_string(index=False))

    # Graphs save karo (test ke liye)
    for model_name, res in results.items():
        fig = plot_confusion_matrix(res['confusion_matrix'], model_name)
        fig.savefig(f"cm_{model_name.replace(' ', '_')}.png", dpi=150)
        print(f"Saved: cm_{model_name.replace(' ', '_')}.png")

    fig_acc = plot_ml_accuracy_comparison(results)
    fig_acc.savefig("ml_accuracy_comparison.png", dpi=150)
    print("Saved: ml_accuracy_comparison.png")

    # Test live prediction
    test_review = "This product is absolutely amazing and I love it so much!"
    result = predict_single_review_ml(test_review, pipelines['Logistic Regression'], 'Logistic Regression')
    print(f"\nLive Prediction Test:")
    print(f"  Review : {test_review}")
    print(f"  Result : {result['label']} (Confidence: {result['confidence']}%)")