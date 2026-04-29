# =============================================================================
# preprocessor.py
# TEXT CLEANING + AUTO COLUMN DETECTION + RULE-BASED FAKE LABEL GENERATION
# =============================================================================

import pandas as pd
import numpy as np
import re
from collections import Counter


# =============================================================================
# STEP 1: AUTO COLUMN DETECTION
# Jab bhi koi bhi dataset upload ho, yeh function automatically
# detect karega ki review column kaunsa hai, rating kaunsa hai, etc.
# =============================================================================

def detect_columns(df):
    """
    Automatically detect which column is review, rating, sentiment, etc.
    Works for ANY dataset — Flipkart, PlayStore, Hotel, Zomato, etc.
    """
    columns_lower = {col.lower(): col for col in df.columns}
    detected = {}

    # Words that disqualify a column from being the review column
    EXCLUDE_WORDS = ['date', 'time', 'url', 'link', 'id', 'city', 'country',
                     'province', 'zip', 'postal', 'source', 'username',
                     'address', 'latitude', 'longitude', 'keys', 'website',
                     'added', 'updated', 'seen', 'title', 'name', 'category']

    # --- Review column ---
    # Exact priority order — first match wins
    review_keywords_priority = [
        'reviews.text',       # Datafiniti hotel exact match
        'review_text',
        'reviewtext',
        'review_body',
        'translated_review',  # PlayStore
        'reviews_text',
        'review',
        'text',
        'comment',
        'content',
        'feedback',
        'body',
        'opinion',
        'remarks',
        'description',
        'message',
    ]

    detected['review_col'] = None

    # Pass 1: Exact match (col name == keyword exactly)
    for kw in review_keywords_priority:
        for col_lower, col_original in columns_lower.items():
            if col_lower == kw:
                detected['review_col'] = col_original
                break
        if detected['review_col']:
            break

    # Pass 2: Substring match — keyword IN col name, but no exclude words
    if detected['review_col'] is None:
        for kw in review_keywords_priority:
            for col_lower, col_original in columns_lower.items():
                if kw in col_lower:
                    # Skip if any exclude word is also in the column name
                    if any(ex in col_lower for ex in EXCLUDE_WORDS):
                        continue
                    # Must be a text column with real length
                    sample = df[col_original].dropna().astype(str).head(20)
                    avg_len = sample.str.len().mean() if len(sample) > 0 else 0
                    if avg_len > 20:
                        detected['review_col'] = col_original
                        break
            if detected['review_col']:
                break

    # Pass 3: Fallback — pick the object column with longest average text
    # that doesn't match any exclude word
    if detected['review_col'] is None:
        best_col, best_len = None, 0
        for col in df.columns:
            col_l = col.lower()
            if any(ex in col_l for ex in EXCLUDE_WORDS):
                continue
            if df[col].dtype == object:
                avg = df[col].dropna().astype(str).str.len().mean()
                if avg and avg > best_len:
                    best_len, best_col = avg, col
        detected['review_col'] = best_col

    # --- Rating column ---
    rating_keywords = ['reviews.rating', 'rating', 'rate', 'score', 'stars', 'star', 'mark', 'overall']
    detected['rating_col'] = None
    # Exact match first
    for kw in rating_keywords:
        for col_lower, col_original in columns_lower.items():
            if col_lower == kw:
                detected['rating_col'] = col_original
                break
        if detected['rating_col']: break
    # Substring match
    if detected['rating_col'] is None:
        for kw in rating_keywords:
            for col_lower, col_original in columns_lower.items():
                if kw in col_lower and 'date' not in col_lower:
                    detected['rating_col'] = col_original
                    break
            if detected['rating_col']: break

    # --- Sentiment column ---
    sentiment_keywords = ['sentiment', 'polarity', 'emotion', 'label', 'class', 'category']
    detected['sentiment_col'] = None
    for kw in sentiment_keywords:
        for col_lower, col_original in columns_lower.items():
            if kw in col_lower and 'fake' not in col_lower and 'primary' not in col_lower:
                detected['sentiment_col'] = col_original
                break
        if detected['sentiment_col']: break

    # --- Fake label column ---
    fake_keywords = ['fake', 'genuine', 'spam', 'fraud', 'authentic']
    detected['fake_col'] = None
    for kw in fake_keywords:
        for col_lower, col_original in columns_lower.items():
            if kw in col_lower:
                detected['fake_col'] = col_original
                break
        if detected['fake_col']: break

    return detected


# =============================================================================
# STEP 2: TEXT CLEANING
# Review text ko clean karna — lowercase, punctuation remove, stopwords remove
# =============================================================================

# Basic stopwords list (NLTK ke bina bhi kaam karega)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
    'wasn', 'weren', 'won', 'wouldn'
])


def clean_text(text):
    """
    Single review text ko clean karo.
    Steps:
    1. Lowercase
    2. Special characters/numbers remove
    3. Extra spaces remove
    4. Stopwords remove
    """
    if not isinstance(text, str) or text.strip() == '':
        return ''

    # Lowercase
    text = text.lower()

    # URLs remove karo
    text = re.sub(r'http\S+|www\S+', '', text)

    # Special characters aur numbers remove karo (sirf letters rakho)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Extra whitespace remove karo
    text = re.sub(r'\s+', ' ', text).strip()

    # Stopwords remove karo
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return ' '.join(words)


def clean_dataset(df, review_col):
    """
    Poore dataset ke review column ko clean karo.
    Ek naya column 'Clean_Review' add karo.

    NOTE: We do NOT drop rows whose Clean_Review becomes empty after
    stopword removal — instead we fall back to the raw review text.
    This prevents ZeroDivisionError on datasets with very short reviews.
    """
    print(f"[INFO] Cleaning text in column: '{review_col}'...")
    df = df.copy()
    df['Clean_Review'] = df[review_col].apply(clean_text)

    # Fallback: agar clean karne ke baad kuch nahi bacha toh original text use karo
    empty_mask = df['Clean_Review'].str.strip() == ''
    empty_count = empty_mask.sum()
    if empty_count > 0:
        # Use lowercased original as fallback so model still has something
        df.loc[empty_mask, 'Clean_Review'] = (
            df.loc[empty_mask, review_col]
            .astype(str)
            .str.lower()
            .str.replace(r'[^a-zA-Z\s]', '', regex=True)
            .str.strip()
        )
        print(f"[INFO] {empty_count} reviews were empty after cleaning — used raw text fallback.")

    # Still drop rows that are completely empty even after fallback
    before = len(df)
    df = df[df['Clean_Review'].str.strip() != ''].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} truly empty rows (no text at all).")

    print(f"[INFO] Text cleaning done. Total reviews: {len(df)}")
    return df


# =============================================================================
# STEP 3: RULE-BASED FAKE LABEL GENERATION
# Yeh sabse important part hai — bina labeled data ke bhi fake detect karo
# =============================================================================

def check_too_short(text, min_chars=15):
    """Rule 1: Review bahut chota hai → FAKE. e.g. 'ok', 'good'"""
    if not isinstance(text, str):
        return True
    return len(text.strip()) < min_chars


def check_repeated_words(text, threshold=0.4):
    """Rule 2: Ek hi word zyada baar repeat hua → FAKE.
    e.g. 'good product good product good product'
    Runs on stopword-removed text so 'the/and/were' don't dilute the ratio.
    Threshold 0.40 = 40% meaningful words same hain toh fake."""
    if not isinstance(text, str) or len(text.split()) < 4:
        return False

    # Remove stopwords before checking — so 'the/and/were' don't hide repetition
    stop = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'was','were','is','are','be','been','being','have','has','had',
        'do','does','did','will','would','could','should','may','might',
        'this','that','these','those','it','its','we','i','my','our',
        'very','so','too','also','just','not','no','nor','yet','both',
        'all','any','each','few','more','most','other','some','such',
        'than','then','when','where','how','what','which','who','whom',
    }
    words = [w for w in text.lower().split() if w not in stop and len(w) > 2]

    if len(words) < 3:
        return False

    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]
    ratio = most_common_count / len(words)
    return ratio >= threshold


def check_duplicate_phrases(text):
    """Rule 3: Same phrase/sentence 2 baar aayi → FAKE.
    e.g. 'Very bad experience Very bad experience'
    Also catches repeating bigrams: 'very nice hotel very nice stay very nice rooms'"""
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    mid = len(t) // 2
    first_half = t[:mid].strip()
    second_half = t[mid:].strip()
    if first_half and (first_half == second_half or t.count(first_half) > 1):
        return True

    words = t.split()

    # Check repeating N-word exact chunks
    if len(words) >= 4:
        for chunk_size in [2, 3, 4]:
            chunks = [' '.join(words[i:i+chunk_size])
                      for i in range(0, len(words) - chunk_size + 1, chunk_size)]
            if len(chunks) >= 2 and len(set(chunks)) == 1:
                return True

    # Check repeating bigrams (sliding window)
    # e.g. "very nice hotel very nice stay very nice rooms" → "very nice" appears 3+ times
    if len(words) >= 6:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        top_bigram, top_count = bigram_counts.most_common(1)[0]
        # If same 2-word combo appears 3+ times → clearly repeating pattern
        if top_count >= 3:
            return True

    return False


def check_rating_sentiment_mismatch(rating, sentiment):
    """Rule 4: Rating aur Sentiment contradict karte hain → FAKE.
    e.g. 5-star + Negative review"""
    if pd.isna(rating) or pd.isna(sentiment):
        return False
    try:
        rating = float(rating)
    except:
        return False
    sentiment = str(sentiment).lower().strip()
    if rating >= 4 and sentiment == 'negative':
        return True
    if rating <= 2 and sentiment == 'positive':
        return True
    return False


def check_generic_only(text):
    """Rule 5: Sirf generic/vague words hain, koi specific detail nahi → FAKE.
    e.g. 'good product nice item great stuff'"""
    generic_words = {
        'good', 'bad', 'ok', 'okay', 'nice', 'fine', 'great', 'awesome',
        'terrible', 'worst', 'best', 'product', 'item', 'thing', 'stuff',
        'excellent', 'amazing', 'wonderful', 'perfect', 'fantastic',
        'horrible', 'poor', 'average', 'decent', 'superb', 'outstanding'
    }
    if not isinstance(text, str):
        return True
    words = set(re.sub(r'[^a-zA-Z\s]', '', text.lower()).split())
    words = {w for w in words if len(w) > 2}
    if not words:
        return True
    non_generic = words - generic_words
    return len(non_generic) == 0


def check_excessive_punctuation(text):
    """Rule 6: Bahut zyada exclamation marks ya caps → FAKE.
    e.g. 'AMAZING!!! BUY NOW!!! BEST EVER!!!'"""
    if not isinstance(text, str):
        return False
    exclamations = text.count('!')
    question_marks = text.count('?')
    # 3+ exclamation marks
    if exclamations >= 3:
        return True
    # ALL CAPS text (more than 5 words all caps)
    words = text.split()
    if len(words) >= 4:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) / len(words) >= 0.6:
            return True
    return False


def check_exaggerated_language(text):
    """Rule 7: Exaggerated/unrealistic claims → FAKE.
    e.g. '1000%', 'best in the world', 'never seen anything like this'
    Also catches: 'very very very', consecutive repeated adjectives"""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()

    # Exaggeration numbers: 100%, 1000%, 200%
    if re.search(r'\b\d{3,}%', text):
        return True

    # Consecutive repeated words: "very very very", "good good good"
    if re.search(r'\b(\w+)\s+\1\s+\1\b', text_lower):
        return True

    # Extreme superlative phrases
    extreme_phrases = [
        'best in the world', 'best ever', 'never seen anything like',
        'must buy', 'must visit', 'dont miss', "don't miss",
        'highly highly', '100% recommend', '1000%', 'go go go',
        'buy buy buy', 'visit visit', 'love love love',
        'absolutely perfect', 'absolutely amazing', 'totally amazing',
        'completely perfect', 'beyond perfect', 'out of this world'
    ]
    for phrase in extreme_phrases:
        if phrase in text_lower:
            return True

    return False


def check_no_specific_details(text):
    """Rule 8: Review mein koi specific detail nahi — sirf vague praise/criticism → FAKE.
    Genuine reviews mention: dates, names, specific features, comparisons.
    We check: review >= 40 words but zero specific nouns/details."""
    if not isinstance(text, str):
        return False
    words = text.lower().split()
    if len(words) < 10:  # Too short reviews caught by Rule 1 already
        return False

    # Specific detail indicators — if ANY of these exist, likely genuine
    specific_indicators = [
        # Room/hotel specifics
        r'\b(room|floor|lobby|pool|gym|spa|wifi|parking|breakfast|dinner|lunch)\b',
        # Product specifics
        r'\b(battery|camera|screen|size|color|colour|weight|material|quality|price)\b',
        # Time references
        r'\b(day|night|week|month|year|hour|morning|evening|monday|tuesday|wednesday|january|february|march)\b',
        # Numbers (specific data)
        r'\b\d+\b',
        # Names of people or places
        r'\b(staff|manager|reception|service|location|area|city|place)\b',
        # Comparisons
        r'\b(than|compared|versus|better|worse|similar|unlike)\b',
    ]

    text_lower = text.lower()
    for pattern in specific_indicators:
        if re.search(pattern, text_lower):
            return False  # Has specific detail → likely genuine

    # No specific details found and review is long enough to be suspicious
    return len(words) >= 15


def generate_fake_labels(df, review_col, rating_col=None, sentiment_col=None):
    """
    Fake_Label column generate karo using 8 smart rules.

    Rule 1: Too short (< 15 chars)
    Rule 2: Repeated words (40%+ same word)
    Rule 3: Duplicate phrase / repeating chunk
    Rule 4: Rating-Sentiment mismatch
    Rule 5: Generic words only (no specific detail)
    Rule 6: Excessive punctuation / ALL CAPS
    Rule 7: Exaggerated language (1000%, very very very, must buy)
    Rule 8: Long review but zero specific details

    Fake_Label: 1 = Fake, 0 = Genuine
    """
    print("[INFO] Generating Fake Labels using 8-rule approach...")
    df = df.copy()

    fake_labels = []
    reasons = []

    for idx, row in df.iterrows():
        review   = str(row.get(review_col, ''))
        rating   = row.get(rating_col)   if rating_col   else None
        sentiment= row.get(sentiment_col) if sentiment_col else None

        is_fake = False
        reason  = []

        if check_too_short(review, min_chars=15):
            is_fake = True; reason.append('too_short')

        if check_repeated_words(review, threshold=0.4):
            is_fake = True; reason.append('repeated_words')

        if check_duplicate_phrases(review):
            is_fake = True; reason.append('duplicate_phrase')

        if rating_col and sentiment_col:
            if check_rating_sentiment_mismatch(rating, sentiment):
                is_fake = True; reason.append('rating_sentiment_mismatch')

        if check_generic_only(review):
            is_fake = True; reason.append('generic_only')

        if check_excessive_punctuation(review):
            is_fake = True; reason.append('excessive_punctuation')

        if check_exaggerated_language(review):
            is_fake = True; reason.append('exaggerated_language')

        if check_no_specific_details(review):
            is_fake = True; reason.append('no_specific_details')

        fake_labels.append(1 if is_fake else 0)
        reasons.append(', '.join(reason) if reason else 'genuine')

    df['Fake_Label'] = fake_labels
    df['Fake_Reason'] = reasons

    fake_count    = sum(fake_labels)
    genuine_count = len(fake_labels) - fake_count
    total         = len(df)

    print(f"[INFO] Fake Label generation done!")
    if total > 0:
        print(f"       Fake reviews    : {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"       Genuine reviews : {genuine_count} ({genuine_count/total*100:.1f}%)")
    else:
        print("       WARNING: No rows remain after preprocessing!")

    return df


# =============================================================================
# STEP 4: MISSING VALUE HANDLING + FINAL PREP
# =============================================================================

def handle_missing_values(df, review_col):
    """
    Missing values handle karo.
    """
    print("[INFO] Handling missing values...")
    before = len(df)

    # Review column mein NaN rows drop karo — review ke bina kuch nahi kar sakte
    df = df.dropna(subset=[review_col]).reset_index(drop=True)

    # Baaki columns mein NaN → 'Unknown' se fill karo
    df = df.fillna('Unknown')

    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing reviews.")

    return df


# =============================================================================
# STEP 5: FULL PIPELINE — Ek function mein sab kuch
# Yeh function app.py mein call karenge
# =============================================================================

def full_preprocessing_pipeline(df, max_rows=15000):
    """
    Complete preprocessing pipeline.
    Input  : Raw uploaded DataFrame (any dataset)
    Output : Cleaned DataFrame with Clean_Review + Fake_Label columns

    max_rows: If dataset has more rows than this, randomly sample to keep
              training fast and avoid memory issues. (Default: 15,000)
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE STARTED")
    print("=" * 50)

    # ── Safety check: empty dataframe ──
    if len(df) == 0:
        raise ValueError("The uploaded dataset is empty. Please upload a valid CSV file.")

    # ── Sample large datasets ──
    original_len = len(df)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"[INFO] Large dataset detected ({original_len:,} rows). "
              f"Sampled {max_rows:,} rows for faster processing.")

    # ── Step 1: Auto detect columns ──
    col_info = detect_columns(df)
    print(f"[INFO] Detected columns: {col_info}")

    review_col    = col_info['review_col']
    rating_col    = col_info['rating_col']
    sentiment_col = col_info['sentiment_col']
    fake_col      = col_info['fake_col']

    if review_col is None:
        raise ValueError(
            "Could not find a review/text column in your dataset. "
            "Please make sure your CSV has a column named 'review', 'text', 'comment', "
            "'feedback', or similar. Columns found: " + str(df.columns.tolist())
        )

    # ── Step 2: Missing values ──
    df = handle_missing_values(df, review_col)

    if len(df) == 0:
        raise ValueError(
            f"After removing missing values, 0 rows remain. "
            f"The review column '{review_col}' may be empty or misdetected. "
            f"Please check your CSV file."
        )

    # ── Step 3: Text cleaning ──
    df = clean_dataset(df, review_col)

    if len(df) == 0:
        raise ValueError(
            "After text cleaning, 0 rows remain. "
            "The review column may contain only numbers or symbols. "
            f"Detected review column: '{review_col}'"
        )

    # ── Step 4: Fake labels ──
    if fake_col is None:
        df = generate_fake_labels(df, review_col, rating_col, sentiment_col)
        col_info['fake_col'] = 'Fake_Label'
    else:
        print(f"[INFO] Fake_Label column '{fake_col}' already exists. Skipping generation.")
        if fake_col != 'Fake_Label':
            df['Fake_Label'] = df[fake_col]
            col_info['fake_col'] = 'Fake_Label'

    # ── Final check: need at least both classes ──
    if df['Fake_Label'].nunique() < 2:
        # Force at least some fake labels so ML can train
        print("[WARN] Only one class detected in Fake_Label. Applying rule-based override...")
        df = generate_fake_labels(df, review_col, rating_col, sentiment_col)
        col_info['fake_col'] = 'Fake_Label'

    print("=" * 50)
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"Final dataset shape: {df.shape}")
    print("=" * 50)

    return df, col_info


# =============================================================================
# TEST — Yeh tab chalega jab tum directly "python preprocessor.py" run karo
# =============================================================================

if __name__ == "__main__":
    # Apna dataset test karo
    df_raw = pd.read_csv('../fake_review_dataset_with_labels.csv')
    print("Original shape:", df_raw.shape)
    print("Columns:", df_raw.columns.tolist())
    print()

    df_clean, col_info = full_preprocessing_pipeline(df_raw)

    print("\nSample output:")
    print(df_clean[['Clean_Review', 'Fake_Label', 'Fake_Reason']].head(10).to_string())