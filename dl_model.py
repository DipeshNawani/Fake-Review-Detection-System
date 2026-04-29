# =============================================================================
# dl_model.py
# DEEP LEARNING PART — LSTM Model for Fake Review Detection
# Embedding → LSTM → Dense → Output
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Embedding, LSTM, Dense, Dropout,
        Bidirectional, GlobalMaxPooling1D
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    print("[DL] TensorFlow loaded successfully.")
except ImportError:
    TF_AVAILABLE = False
    print("[DL] WARNING: TensorFlow not found. Install it with: pip install tensorflow")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)


# =============================================================================
# STEP 1: TOKENIZATION + PADDING
# Text ko numbers mein convert karo for LSTM
# =============================================================================

# Hyperparameters — Examiner ko yeh zaoor batana
MAX_VOCAB_SIZE  = 10000   # Vocabulary size — top 10k words
MAX_SEQ_LENGTH  = 100     # Har review max 100 words tak
EMBEDDING_DIM   = 64      # Word embedding dimension
LSTM_UNITS      = 64      # LSTM ke units
DROPOUT_RATE    = 0.3     # Overfitting rokne ke liye
BATCH_SIZE      = 32      # Ek baar mein kitne samples process ho
EPOCHS          = 10      # Kitni baar poora dataset train ho


def tokenize_and_pad(X_train, X_test):
    """
    Text ko integer sequences mein convert karo + pad karo.
    
    LSTM ko fixed length input chahiye → isliye padding karte hain.
    
    Returns:
        X_train_pad, X_test_pad : Padded sequences
        tokenizer               : Fit tokenizer (prediction ke liye chahiye)
    """
    print("[DL] Tokenizing and padding sequences...")

    # Tokenizer banao — har word ko ek number assign karega
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token='<OOV>'  # Out-of-vocabulary words ke liye
    )

    # Training data pe fit karo (test pe nahi — data leakage rokne ke liye)
    tokenizer.fit_on_texts(X_train)

    # Text → Integer sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    # Padding — sab sequences same length banao
    # 'post' → end mein zeros add karo
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

    vocab_size = len(tokenizer.word_index) + 1
    print(f"[DL] Vocabulary size : {vocab_size}")
    print(f"[DL] Sequence length : {MAX_SEQ_LENGTH}")
    print(f"[DL] Train shape     : {X_train_pad.shape}")
    print(f"[DL] Test shape      : {X_test_pad.shape}")

    return X_train_pad, X_test_pad, tokenizer


# =============================================================================
# STEP 2: LSTM MODEL ARCHITECTURE
# Embedding → Bidirectional LSTM → Dropout → Dense → Output
# =============================================================================

def build_lstm_model(vocab_size):
    """
    Bidirectional LSTM model build karo.
    
    Architecture:
    1. Embedding Layer    → Words ko dense vectors mein convert karo
    2. Bidirectional LSTM → Forward + Backward dono directions mein padhta hai
    3. Dropout            → Overfitting rokta hai
    4. Dense (ReLU)       → Feature extraction
    5. Dropout            → Aur regularization
    6. Dense (Sigmoid)    → Binary output: Fake (1) ya Genuine (0)
    
    Loss     : Binary Crossentropy (binary classification ke liye standard)
    Optimizer: Adam (adaptive learning rate)
    Metric   : Accuracy
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    print("[DL] Building LSTM model architecture...")

    model = Sequential([
        # Layer 1: Embedding
        # input_dim  = vocabulary size
        # output_dim = embedding dimension (word vector size)
        # input_length = sequence length
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQ_LENGTH,
            name='embedding_layer'
        ),

        # Layer 2: Bidirectional LSTM
        # Bidirectional → dono directions (left-to-right + right-to-left) padhta hai
        # return_sequences=False → sirf last output chahiye
        Bidirectional(
            LSTM(LSTM_UNITS, return_sequences=False),
            name='bidirectional_lstm'
        ),

        # Layer 3: Dropout (Regularization)
        Dropout(DROPOUT_RATE, name='dropout_1'),

        # Layer 4: Dense layer with ReLU activation
        Dense(32, activation='relu', name='dense_hidden'),

        # Layer 5: Another Dropout
        Dropout(DROPOUT_RATE, name='dropout_2'),

        # Layer 6: Output layer
        # 1 neuron + Sigmoid → probability between 0 and 1
        # > 0.5 → FAKE (1), < 0.5 → GENUINE (0)
        Dense(1, activation='sigmoid', name='output_layer')
    ])

    # Compile karo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("[DL] Model architecture:")
    model.summary()

    return model


# =============================================================================
# STEP 3: TRAINING
# =============================================================================

def train_lstm(df, text_col='Clean_Review', label_col='Fake_Label'):
    """
    Full LSTM training pipeline.
    
    Returns:
        history    : Training history (accuracy/loss curves ke liye)
        model      : Trained Keras model
        tokenizer  : Fitted tokenizer (prediction ke liye)
        results    : Evaluation metrics dict
        X_test_pad, y_test : Testing data (confusion matrix ke liye)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    print("\n" + "=" * 50)
    print("DL MODEL TRAINING STARTED")
    print("=" * 50)

    # Data prepare
    X = df[text_col].fillna('').values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[DL] Training samples : {len(X_train)}")
    print(f"[DL] Testing  samples : {len(X_test)}")

    # Tokenize + Pad
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad(X_train, X_test)

    # Vocabulary size
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    # Model build karo
    model = build_lstm_model(vocab_size)

    # Class weights calculate karo (imbalanced dataset ke liye)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
    print(f"[DL] Class weights: {class_weight_dict}")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,           # 3 epochs tak improvement nahi → stop
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # Learning rate half karo
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # TRAIN!
    print(f"\n[DL] Training for {EPOCHS} epochs (max), batch_size={BATCH_SIZE}...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,   # 10% training data → validation ke liye
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    # EVALUATE
    print("\n[DL] Evaluating on test data...")
    y_pred_prob = model.predict(X_test_pad, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'], zero_division=0)

    results = {
        'accuracy':              round(acc * 100, 2),
        'precision':             round(precision * 100, 2),
        'recall':                round(recall * 100, 2),
        'f1_score':              round(f1 * 100, 2),
        'confusion_matrix':      cm,
        'classification_report': report,
        'y_test':                y_test,
        'y_pred':                y_pred,
        'y_pred_prob':           y_pred_prob.flatten()
    }

    print(f"\n  ✅ LSTM Accuracy  : {acc*100:.2f}%")
    print(f"     Precision      : {precision*100:.2f}%")
    print(f"     Recall         : {recall*100:.2f}%")
    print(f"     F1-Score       : {f1*100:.2f}%")
    print("\n" + "=" * 50)
    print("DL MODEL TRAINING COMPLETE")
    print("=" * 50)

    return history, model, tokenizer, results


# =============================================================================
# STEP 4: VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """
    Training history plot karo — Accuracy aur Loss curves.
    Examiner ke liye bahut important graph hai.
    Returns: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs_ran = len(history.history['accuracy'])
    epoch_range = range(1, epochs_ran + 1)

    # --- Plot 1: Accuracy ---
    axes[0].plot(epoch_range, history.history['accuracy'],
                 'b-o', label='Training Accuracy', linewidth=2, markersize=5)
    axes[0].plot(epoch_range, history.history['val_accuracy'],
                 'r-s', label='Validation Accuracy', linewidth=2, markersize=5)
    axes[0].set_title('Model Accuracy over Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(epoch_range)

    # --- Plot 2: Loss ---
    axes[1].plot(epoch_range, history.history['loss'],
                 'b-o', label='Training Loss', linewidth=2, markersize=5)
    axes[1].plot(epoch_range, history.history['val_loss'],
                 'r-s', label='Validation Loss', linewidth=2, markersize=5)
    axes[1].set_title('Model Loss over Epochs', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(epoch_range)

    plt.suptitle('LSTM Training History', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_dl_confusion_matrix(cm):
    """
    LSTM ka confusion matrix plot karo.
    Returns: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=['Genuine', 'Fake'],
        yticklabels=['Genuine', 'Fake'],
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('Confusion Matrix — LSTM (Bidirectional)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# STEP 5: LIVE PREDICTION USING TRAINED LSTM
# =============================================================================

def predict_single_review_dl(review_text, model, tokenizer):
    """
    Single review ke liye LSTM prediction.
    
    Input : Review text string
    Output: {label, confidence}
    """
    from preprocessor import clean_text

    # Clean karo
    cleaned = clean_text(review_text)

    if not cleaned:
        return {'label': 'Unable to predict', 'confidence': 0}

    # Tokenize + Pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

    # Predict
    prob = model.predict(padded, verbose=0)[0][0]
    label = 'FAKE' if prob > 0.5 else 'GENUINE'
    confidence = round(float(prob) * 100 if prob > 0.5 else (1 - float(prob)) * 100, 1)

    return {
        'label': label,
        'confidence': confidence,
        'raw_probability': round(float(prob), 4),
        'cleaned_review': cleaned
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    if not TF_AVAILABLE:
        print("Install TensorFlow first: pip install tensorflow")
        exit()

    from preprocessor import full_preprocessing_pipeline

    df_raw = pd.read_csv('../fake_review_dataset_with_labels.csv')
    df_clean, col_info = full_preprocessing_pipeline(df_raw)

    history, model, tokenizer, results = train_lstm(df_clean)

    print(f"\nFinal LSTM Accuracy: {results['accuracy']}%")
    print("\nClassification Report:")
    print(results['classification_report'])

    # Save plots
    fig1 = plot_training_history(history)
    fig1.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
    print("Saved: lstm_training_history.png")

    fig2 = plot_dl_confusion_matrix(results['confusion_matrix'])
    fig2.savefig('lstm_confusion_matrix.png', dpi=150)
    print("Saved: lstm_confusion_matrix.png")

    # Live prediction test
    test_reviews = [
        "This product is absolutely amazing and I love it so much!",
        "bad bad bad bad bad",
        "Good product good product good product"
    ]
    print("\nLive Prediction Tests:")
    for rev in test_reviews:
        result = predict_single_review_dl(rev, model, tokenizer)
        print(f"  '{rev[:50]}...' → {result['label']} ({result['confidence']}%)")