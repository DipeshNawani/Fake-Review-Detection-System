# =============================================================================
# visualizations.py
# ALL EDA VISUALIZATIONS — Pie Chart, WordCloud, Distribution Graphs,
# ML vs DL Comparison Chart
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import re

# Seaborn style
sns.set_theme(style="whitegrid", palette="muted")


# =============================================================================
# 1. FAKE VS GENUINE PIE CHART
# =============================================================================

def plot_fake_genuine_pie(df, label_col='Fake_Label'):
    """
    Fake vs Genuine distribution pie chart.
    Examiner ka first impression yahi se hoga!
    """
    counts = df[label_col].value_counts()
    labels = ['Genuine (0)', 'Fake (1)']
    sizes  = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 12}
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    ax.set_title('Fake vs Genuine Review Distribution', fontsize=14, fontweight='bold', pad=20)

    # Counts bhi dikhao
    legend_labels = [f'Genuine: {sizes[0]:,}', f'Fake: {sizes[1]:,}']
    ax.legend(legend_labels, loc='lower right', fontsize=11)

    plt.tight_layout()
    return fig


# =============================================================================
# 2. SENTIMENT DISTRIBUTION BAR CHART
# =============================================================================

def plot_sentiment_distribution(df, sentiment_col='Sentiment'):
    """
    Sentiment distribution — Positive / Negative / Neutral.
    """
    if sentiment_col not in df.columns:
        return None

    sentiment_counts = df[sentiment_col].value_counts()
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    bar_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                  color=bar_colors, edgecolor='white', linewidth=0.7)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{int(bar.get_height()):,}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


# =============================================================================
# 3. RATING DISTRIBUTION
# =============================================================================

def plot_rating_distribution(df, rating_col='Rate'):
    """
    Rating distribution (1-5 stars).
    Fake vs Genuine ke liye alag alag dikhao.
    """
    if rating_col not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall rating distribution
    rating_counts = df[rating_col].value_counts().sort_index()
    axes[0].bar(rating_counts.index.astype(str), rating_counts.values,
                color='#3498db', edgecolor='white', linewidth=0.7)
    axes[0].set_title('Overall Rating Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Rating (Stars)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Rating by Fake/Genuine
    if 'Fake_Label' in df.columns:
        genuine_ratings = df[df['Fake_Label'] == 0][rating_col].value_counts().sort_index()
        fake_ratings    = df[df['Fake_Label'] == 1][rating_col].value_counts().sort_index()

        x = np.arange(1, 6)
        width = 0.35
        axes[1].bar(x - width/2,
                    [genuine_ratings.get(i, 0) for i in x],
                    width, label='Genuine', color='#2ecc71', edgecolor='white')
        axes[1].bar(x + width/2,
                    [fake_ratings.get(i, 0) for i in x],
                    width, label='Fake', color='#e74c3c', edgecolor='white')
        axes[1].set_title('Rating Distribution: Fake vs Genuine', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Rating (Stars)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_xticks(x)
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# 4. WORDCLOUD — FAKE VS GENUINE
# =============================================================================

def get_top_words(df, label_col='Fake_Label', text_col='Clean_Review', label_value=1, top_n=50):
    """
    Fake ya Genuine reviews ke top words nikalo.
    WordCloud ke badle simple frequency bar chart banate hain
    (wordcloud library ke bina bhi kaam kare).
    """
    subset = df[df[label_col] == label_value][text_col].fillna('')
    all_text = ' '.join(subset.values)
    words = all_text.split()
    word_counts = Counter(words)
    # Very common generic words bhi hata do
    skip_words = {'product', 'item', 'one', 'use', 'get', 'also', 'would', 'really', 'buy'}
    top_words = [(w, c) for w, c in word_counts.most_common(top_n + 20)
                 if w not in skip_words][:top_n]
    return top_words


def plot_top_words_comparison(df, text_col='Clean_Review', label_col='Fake_Label', top_n=15):
    """
    Top words in Fake reviews vs Genuine reviews — side by side bar chart.
    WordCloud library ke bina bhi perfectly kaam karta hai.
    """
    fake_words    = get_top_words(df, label_col, text_col, label_value=1, top_n=top_n)
    genuine_words = get_top_words(df, label_col, text_col, label_value=0, top_n=top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fake words
    if fake_words:
        words_f, counts_f = zip(*fake_words)
        axes[0].barh(list(reversed(words_f)), list(reversed(counts_f)),
                     color='#e74c3c', edgecolor='white', linewidth=0.5)
        axes[0].set_title('Top Words in FAKE Reviews', fontsize=13, fontweight='bold', color='#e74c3c')
        axes[0].set_xlabel('Frequency', fontsize=11)
        axes[0].grid(axis='x', linestyle='--', alpha=0.4)

    # Genuine words
    if genuine_words:
        words_g, counts_g = zip(*genuine_words)
        axes[1].barh(list(reversed(words_g)), list(reversed(counts_g)),
                     color='#2ecc71', edgecolor='white', linewidth=0.5)
        axes[1].set_title('Top Words in GENUINE Reviews', fontsize=13, fontweight='bold', color='#2ecc71')
        axes[1].set_xlabel('Frequency', fontsize=11)
        axes[1].grid(axis='x', linestyle='--', alpha=0.4)

    plt.suptitle('Word Frequency Analysis: Fake vs Genuine Reviews',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# =============================================================================
# 5. FAKE REASON DISTRIBUTION
# Konsa rule kitne reviews pe trigger hua
# =============================================================================

def plot_fake_reasons(df):
    """
    Fake_Reason column se — konsi rule kitni baar trigger hui.
    """
    if 'Fake_Reason' not in df.columns:
        return None

    fake_df = df[df['Fake_Label'] == 1].copy()
    if fake_df.empty:
        return None

    # Reasons expand karo
    all_reasons = []
    for reasons_str in fake_df['Fake_Reason']:
        if reasons_str and reasons_str != 'genuine':
            for r in reasons_str.split(','):
                all_reasons.append(r.strip())

    if not all_reasons:
        return None

    reason_counts = Counter(all_reasons)

    # Readable names
    readable = {
        'too_short':                  'Too Short Review',
        'repeated_words':             'Repeated Words',
        'duplicate_phrase':           'Duplicate Phrase',
        'rating_sentiment_mismatch':  'Rating-Sentiment Mismatch',
        'generic_only':               'Generic Words Only'
    }

    labels  = [readable.get(r, r) for r in reason_counts.keys()]
    values  = list(reason_counts.values())
    colors  = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#3498db']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=0.5)

    for bar in bars:
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2.,
                f'{int(bar.get_width()):,}', va='center', fontsize=10, fontweight='bold')

    ax.set_title('Why Reviews Were Marked as FAKE\n(Rule-Based Trigger Frequency)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Reviews Triggered', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =============================================================================
# 6. REVIEW LENGTH ANALYSIS
# =============================================================================

def plot_review_length_analysis(df, review_col='Review', label_col='Fake_Label'):
    """
    Fake vs Genuine reviews ki length distribution.
    Usually fake reviews → shorter hote hain.
    """
    if review_col not in df.columns or label_col not in df.columns:
        return None

    df = df.copy()
    df['review_length'] = df[review_col].fillna('').apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    genuine_lengths = df[df[label_col] == 0]['review_length']
    fake_lengths    = df[df[label_col] == 1]['review_length']

    axes[0].hist(genuine_lengths, bins=30, alpha=0.7, color='#2ecc71', label='Genuine', edgecolor='white')
    axes[0].hist(fake_lengths,    bins=30, alpha=0.7, color='#e74c3c', label='Fake',    edgecolor='white')
    axes[0].set_title('Review Length Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Words', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # Box plot
    data_to_plot = [genuine_lengths.values, fake_lengths.values]
    bp = axes[1].boxplot(data_to_plot, patch_artist=True, notch=False,
                         labels=['Genuine', 'Fake'])
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_title('Review Length Box Plot', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Words', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Review Length: Fake vs Genuine', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# =============================================================================
# 7. ML vs DL FINAL COMPARISON CHART
# Sabse important chart — examiner yahi dekhega
# =============================================================================

def plot_ml_dl_comparison(ml_results, dl_results):
    """
    ML models + LSTM accuracy/F1 ek saath compare karo.
    
    ml_results : dict from ml_model.py → train_and_evaluate_all()
    dl_results : dict from dl_model.py → train_lstm() results
    """
    all_models = {}
    for name, res in ml_results.items():
        all_models[name] = res

    all_models['LSTM (DL)'] = dl_results

    model_names = list(all_models.keys())
    accuracies  = [all_models[m]['accuracy']  for m in model_names]
    f1_scores   = [all_models[m]['f1_score']  for m in model_names]

    # Colors: ML models blue shades, DL model green
    colors_acc = ['#4C72B0', '#4C72B0', '#4C72B0', '#27ae60']
    colors_f1  = ['#DD8452', '#DD8452', '#DD8452', '#f39c12']

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)',
                   color=colors_acc, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)',
                   color=colors_f1, edgecolor='white', linewidth=0.7)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # LSTM ko highlight karo
    ax.axvline(x=len(model_names) - 1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(model_names) - 1.5, max(accuracies) + 3,
            '← ML  |  DL →', ha='center', fontsize=9, color='gray')

    ax.set_title('ML vs Deep Learning — Final Model Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_summary_metrics_table(ml_results, dl_results):
    """
    Final summary table — sabhi models ke metrics.
    Returns: pandas DataFrame
    """
    rows = []
    for model_name, res in ml_results.items():
        rows.append({
            'Model': model_name,
            'Type': 'Machine Learning',
            'Accuracy (%)': res['accuracy'],
            'Precision (%)': res['precision'],
            'Recall (%)': res['recall'],
            'F1-Score (%)': res['f1_score']
        })
    rows.append({
        'Model': 'LSTM (Bidirectional)',
        'Type': 'Deep Learning',
        'Accuracy (%)': dl_results['accuracy'],
        'Precision (%)': dl_results['precision'],
        'Recall (%)': dl_results['recall'],
        'F1-Score (%)': dl_results['f1_score']
    })
    return pd.DataFrame(rows)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessor import full_preprocessing_pipeline

    df_raw = pd.read_csv('../fake_review_dataset_with_labels.csv')
    df_clean, col_info = full_preprocessing_pipeline(df_raw)

    # Test each plot
    fig1 = plot_fake_genuine_pie(df_clean)
    fig1.savefig('test_pie.png', dpi=150)
    print("Saved test_pie.png")

    fig2 = plot_sentiment_distribution(df_clean)
    if fig2:
        fig2.savefig('test_sentiment.png', dpi=150)
        print("Saved test_sentiment.png")

    fig3 = plot_top_words_comparison(df_clean)
    fig3.savefig('test_wordfreq.png', dpi=150)
    print("Saved test_wordfreq.png")

    fig4 = plot_fake_reasons(df_clean)
    if fig4:
        fig4.savefig('test_reasons.png', dpi=150)
        print("Saved test_reasons.png")

    fig5 = plot_review_length_analysis(df_clean)
    if fig5:
        fig5.savefig('test_length.png', dpi=150)
        print("Saved test_length.png")