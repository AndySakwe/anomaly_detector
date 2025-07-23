import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER, "rf_model.joblib")
ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.joblib")
CONF_MATRIX_PATH = os.path.join(MODEL_FOLDER, "confusion_matrix.png")

PER_FILE_SAMPLE = 10000
SAMPLE_BENIGN = 10000
SAMPLE_ANOMALY = 10000

def find_label_column(df):
    for col in df.columns:
        if 'label' in col.lower():
            return col
    return None

def load_all_data(folder, per_file_rows):
    data_frames = []
    print("🔍 Scanning for CSV files...\n")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            try:
                print(f"📄 Loading {per_file_rows} rows from {file}...")
                df = pd.read_csv(file_path, nrows=per_file_rows, low_memory=False)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                label_col = find_label_column(df)
                if label_col:
                    labels = pd.read_csv(file_path, usecols=[label_col], nrows=len(df), low_memory=False)
                    df = df.select_dtypes(include='number')
                    df["Label"] = labels[label_col].values
                    data_frames.append(df)
                    print(f"✅ Accepted: {file}")
                else:
                    print(f"⚠️ Skipping {file} (no label column found)")
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")

    if not data_frames:
        raise ValueError("No valid CSV files found.")

    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"\n📊 Total combined dataset size: {len(combined_df)} rows")
    return combined_df

def sample_data(df):
    df["Label"] = df["Label"].astype(str).str.upper().str.strip()
    df["Label"] = df["Label"].apply(lambda x: "BENIGN" if "BENIGN" in x else "ANOMALY")

    benign_df = df[df["Label"] == "BENIGN"]
    anomaly_df = df[df["Label"] == "ANOMALY"]

    sampled_benign = benign_df.sample(n=min(SAMPLE_BENIGN, len(benign_df)), random_state=42)
    sampled_anomaly = anomaly_df.sample(n=min(SAMPLE_ANOMALY, len(anomaly_df)), random_state=42)

    df_sampled = pd.concat([sampled_benign, sampled_anomaly], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1, random_state=42)  # shuffle

    print(f"🎯 Sampled dataset: {len(df_sampled)} rows (Benign: {len(sampled_benign)}, Anomaly: {len(sampled_anomaly)})")
    return df_sampled

def prepare_data(df):
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop('Label', axis=1)
    y = df['Label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    return (X_train, X_test, y_train, y_test, scaler, list(X.columns)), label_encoder

def train_model(X_train, X_test, y_train, y_test, label_encoder):
    print("\n⚙️ Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()

    print(f"📸 Confusion matrix saved to: {CONF_MATRIX_PATH}")
    return rf_model

def main():
    df = load_all_data(DATA_FOLDER, PER_FILE_SAMPLE)
    df = sample_data(df)

    if "Label" not in df.columns:
        raise ValueError("Label column missing after data loading.")

    (X_train, X_test, y_train, y_test, scaler, feature_columns), label_encoder = prepare_data(df)
    rf_model = train_model(X_train, X_test, y_train, y_test, label_encoder)

    # Train Isolation Forest on full scaled data
    print("🌲 Training Isolation Forest...")
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(pd.concat([X_train, X_test]))

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # Save models and utilities
    joblib.dump(rf_model, os.path.join(MODEL_FOLDER, "rf_model.joblib"))
    joblib.dump(iforest, os.path.join(MODEL_FOLDER, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_FOLDER, "feature_columns.pkl"))
    joblib.dump(label_encoder, ENCODER_PATH)

    print("\n✅ All models and artifacts saved to 'models' folder.")
    print("🏁 Training complete. Ready for prediction!")

if __name__ == "__main__":
    main()
