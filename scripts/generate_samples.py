import os
import pandas as pd
import numpy as np

# === CONFIG ===
DATA_FOLDER = "../data"
SAMPLE_OUTPUT_PATH = "../samples/sample.csv"
TOGGLE_PATH = "../samples/.last_sample_type.txt"
SAMPLE_TYPE_ORDER = ["ANOMALY", "BENIGN"]

def find_label_column(df):
    for col in df.columns:
        if 'label' in col.lower():
            return col
    return None

def load_all_data(folder):
    data_frames = []
    print("🔍 Looking for CSV files to sample from...\n")

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            try:
                print(f"📄 Loading {file}...")
                df = pd.read_csv(file_path, low_memory=False, nrows=100000)

                label_col = find_label_column(df)
                if not label_col:
                    print(f"⚠️ Skipped {file} (no label column)")
                    continue

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                df_numeric = df.select_dtypes(include='number').copy()
                df_numeric['Label'] = df[label_col].astype(str).str.upper()
                data_frames.append(df_numeric)
                print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def get_next_sample_type():
    # Default to BENIGN on first run
    if not os.path.exists(TOGGLE_PATH):
        return "BENIGN"

    with open(TOGGLE_PATH, "r") as f:
        last_type = f.read().strip().upper()

    # Flip to the next type
    next_type = "ANOMALY" if last_type == "BENIGN" else "BENIGN"
    return next_type

def save_sample_type(sample_type):
    with open(TOGGLE_PATH, "w") as f:
        f.write(sample_type)

def generate_single_sample(df, sample_type):
    print(f"\n🎯 Generating 1 {sample_type} sample...")
    if sample_type == "BENIGN":
        filtered = df[df['Label'] == 'BENIGN']
    else:
        filtered = df[df['Label'] != 'BENIGN']
        filtered['Label'] = 'ANOMALY'

    if filtered.empty:
        print(f"❌ No data available for {sample_type} sample.")
        return None

    sample = filtered.sample(n=1, random_state=np.random.randint(0, 10000))
    print(f"✅ Sample generated.")
    return sample

def main():
    print("🚀 Starting toggled sample generation...\n")

    df = load_all_data(DATA_FOLDER)
    if df.empty:
        print("⚠️ No valid data found.")
        return

    sample_type = get_next_sample_type()
    sample_df = generate_single_sample(df, sample_type)
    if sample_df is None:
        return

    os.makedirs(os.path.dirname(SAMPLE_OUTPUT_PATH), exist_ok=True)
    sample_df.to_csv(SAMPLE_OUTPUT_PATH, index=False)

    save_sample_type(sample_type)

    print(f"\n📁 {sample_type} sample saved to {SAMPLE_OUTPUT_PATH}")
    print("🎉 Done.")

if __name__ == "__main__":
    main()
