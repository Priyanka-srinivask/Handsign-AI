import os
import warnings
import numpy as np
import pandas as pd
import pickle
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from keras import layers
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SHARED NORMALIZATION — must match inference.py exactly!
# ══════════════════════════════════════════════════════════════
def normalize_landmarks(X):
    """Center on wrist, scale by palm size — position/scale invariant."""
    X_norm = np.zeros_like(X, dtype=float)
    for i in range(len(X)):
        lm        = np.array(X[i], dtype=float).reshape(21, 3)
        wrist     = lm[0]
        centered  = lm - wrist
        palm_size = np.linalg.norm(centered[9])
        if palm_size < 0.001:
            palm_size = 0.001
        X_norm[i] = (centered / palm_size).flatten()
    return X_norm

# ══════════════════════════════════════════════════════════════
# PART 1 — STATIC MODEL (MLP)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 1 — Static Model (MLP Neural Network)")
print("="*60)

STATIC_CSV = "data/static_landmarks.csv"

if not os.path.exists(STATIC_CSV):
    print("⚠️  static_landmarks.csv not found — skipping")
else:
    df_s = pd.read_csv(STATIC_CSV)
    print(f"✅ Loaded {len(df_s)} static samples")
    print(f"\nSamples per sign:")
    print(df_s['label'].value_counts().sort_index())

    FEATURE_COLS = [f'{c}{i}' for i in range(21) for c in ['x','y','z']]
    X_raw = df_s[FEATURE_COLS].values
    y_raw = df_s['label'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y_raw)
    print(f"\nClasses ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    # Split BEFORE normalization
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_encoded,
        test_size=0.2, stratify=y_encoded, random_state=42
    )
    print(f"Train: {len(X_tr_raw)} | Test: {len(X_te_raw)}")

    # Normalize
    print("\nNormalizing landmarks...")
    X_tr_norm = normalize_landmarks(X_tr_raw)
    X_te_norm = normalize_landmarks(X_te_raw)

    # Scale
    scaler   = MinMaxScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_norm)
    X_te_sc  = scaler.transform(X_te_norm)

    # Augment — 10x copies with noise + scale + rotation
    print("Augmenting training data (10x)...")
    def augment_sample(x, noise=0.02, scale_range=(0.9,1.1), rot=0.1):
        lm = x.reshape(21, 3)
        lm = lm + np.random.normal(0, noise, lm.shape)
        lm = lm * np.random.uniform(*scale_range)
        a  = np.random.uniform(-rot, rot)
        R  = np.array([[np.cos(a),-np.sin(a),0],
                       [np.sin(a), np.cos(a),0],
                       [0,0,1]])
        return (lm @ R.T).flatten()

    aug_X, aug_y = [], []
    for i in range(len(X_tr_sc)):
        for _ in range(10):
            aug_X.append(augment_sample(X_tr_sc[i]))
            aug_y.append(y_tr[i])

    X_tr_final = np.vstack([X_tr_sc, np.array(aug_X)])
    y_tr_final = np.concatenate([y_tr, np.array(aug_y)])
    print(f"Training size after augmentation: {len(X_tr_final)}")

    # Build MLP
    NUM_STATIC = len(label_encoder.classes_)
    model_static = keras.Sequential([
        layers.Input(shape=(63,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_STATIC, activation='softmax')
    ])
    model_static.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model_static.summary()

    print("\nTraining Static MLP...")
    t0 = time.time()
    history_s = model_static.fit(
        X_tr_final, y_tr_final,
        validation_data=(X_te_sc, y_te),
        epochs=150,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                                 restore_best_weights=True, verbose=1)],
        verbose=1
    )
    elapsed = time.time() - t0

    _, acc = model_static.evaluate(X_te_sc, y_te, verbose=0)
    y_pred = model_static.predict(X_te_sc, verbose=0).argmax(axis=1)
    print(f"\n✅ Static Accuracy: {acc:.2%} ({elapsed:.0f}s)")
    print(classification_report(y_te, y_pred,
                                target_names=label_encoder.classes_))

    # Save
    model_static.save("models/static_model.keras")
    joblib.dump(scaler,        "models/static_scaler.pkl")
    joblib.dump(label_encoder, "models/static_label_encoder.pkl")
    joblib.dump({
        'classes'             : list(label_encoder.classes_),
        'normalize_landmarks' : True,
        'normalization_method': 'wrist_centered_palm_scaled'
    }, "models/static_config.pkl")
    print("✅ Saved → static_model.keras + scaler + encoder + config")

# ══════════════════════════════════════════════════════════════
# PART 2 — MOTION MODEL (LSTM)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2 — Motion Model (LSTM)")
print("="*60)

if not os.path.exists("data/motion_landmarks.csv"):
    print("⚠️  motion_landmarks.csv not found — skipping")
else:
    df_m = pd.read_csv("data/motion_landmarks.csv")
    print(f"✅ Loaded {len(df_m)} motion rows")

    motion_labels_list = sorted(df_m["label"].unique())
    label_to_idx       = {l: i for i, l in enumerate(motion_labels_list)}
    NUM_MOTION         = len(motion_labels_list)

    print(f"\n{NUM_MOTION} motion signs:")
    print(df_m.groupby('label')['sequence_id'].nunique().sort_index())

    # Build sequences
    feat_cols = [c for c in df_m.columns
                 if c not in ["label","sequence_id","frame_num"]]
    sequences, labels = [], []
    skipped = 0
    for seq_id, grp in df_m.groupby("sequence_id"):
        grp = grp.sort_values("frame_num")
        seq = grp[feat_cols].values.astype(np.float32)
        if seq.shape[0] == 30:
            sequences.append(seq)
            labels.append(label_to_idx[grp["label"].iloc[0]])
        else:
            skipped += 1

    if skipped:
        print(f"⚠️  Skipped {skipped} incomplete sequences")

    X_m = np.array(sequences, dtype=np.float32)
    y_m = to_categorical(labels, NUM_MOTION)
    print(f"\nX shape: {X_m.shape}")
    print(f"y shape: {y_m.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_m, y_m, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # Smart augmentation
    print("\nApplying smart augmentation (6x)...")
    def augment_sequences(X, y, factor=3):
        aug_X, aug_y = [X], [y]
        h1   = (np.abs(X[:,:,0:63]).sum(axis=2, keepdims=True) > 0.01)
        h2   = (np.abs(X[:,:,63:126]).sum(axis=2, keepdims=True) > 0.01)
        mask = np.concatenate([
            np.tile(h1,(1,1,63)),
            np.tile(h2,(1,1,63))
        ], axis=2).astype(np.float32)

        for _ in range(factor):
            # 1. Noise on active landmarks
            noise = np.random.normal(0,0.012,X.shape).astype(np.float32)
            aug_X.append(X + noise * mask); aug_y.append(y)

            # 2. Time shift
            s = np.random.randint(1,4)
            aug_X.append(np.roll(X,s,axis=1)); aug_y.append(y)

            # 3. Speed up
            fast = np.repeat(X[:,::2,:],2,axis=1)[:,:30,:]
            aug_X.append(fast); aug_y.append(y)

            # 4. Slow down
            slow = np.repeat(X,2,axis=1)[:,:30,:]
            aug_X.append(slow); aug_y.append(y)

            # 5. Scale
            sc = np.random.uniform(0.85,1.15)
            aug_X.append(X*sc*mask + X*(1-mask)); aug_y.append(y)

            # 6. Mirror
            mir = X.copy()
            mir[:,:,0:63:3] = (1.0-X[:,:,0:63:3])*mask[:,:,0:63:3] + \
                               X[:,:,0:63:3]*(1-mask[:,:,0:63:3])
            aug_X.append(mir); aug_y.append(y)

        return np.vstack(aug_X), np.vstack(aug_y)

    X_tr, y_tr = augment_sequences(X_tr, y_tr, factor=3)
    idx = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]
    print(f"After augmentation : {len(X_tr)} sequences")
    print(f"Test set unchanged : {len(X_te)} sequences")

    # Build LSTM
    model_motion = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(30, len(feat_cols))),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(NUM_MOTION, activation='softmax')
    ])
    model_motion.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model_motion.summary()

    print(f"\nTraining LSTM...")
    print(f"Batch size: 64 | Max epochs: 100 | Early stop: patience=15\n")

    t0      = time.time()
    history = model_motion.fit(
        X_tr, y_tr,
        epochs=100,
        batch_size=64,
        validation_data=(X_te, y_te),
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=7, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )
    elapsed  = time.time() - t0
    n_epochs = len(history.history['loss'])
    _, acc   = model_motion.evaluate(X_te, y_te, verbose=0)

    print(f"\n✅ Motion Accuracy : {acc:.2%}")
    print(f"   Epochs run      : {n_epochs}/100")
    print(f"   Time taken      : {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save both formats
    model_motion.save("models/motion_model.keras")
    model_motion.save("models/motion_model.h5")
    with open("models/motion_labels.pkl","wb") as f:
        pickle.dump(motion_labels_list, f)
    print("✅ Saved → motion_model.keras + .h5 + motion_labels.pkl")

# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🎉 Training Complete!")
print("="*60)
print("Static : models/static_model.keras")
print("         models/static_scaler.pkl")
print("         models/static_label_encoder.pkl")
print("         models/static_config.pkl")
print("Motion : models/motion_model.keras")
print("         models/motion_model.h5")
print("         models/motion_labels.pkl")