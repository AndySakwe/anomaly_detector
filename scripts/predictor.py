import pandas as pd
import numpy as np
import joblib
import argparse
import os
import time
from sklearn.metrics import accuracy_score

def load_artifacts(model_dir="../models"):
    try:
        rf_model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))
        iforest = joblib.load(os.path.join(model_dir, "isolation_forest.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        features = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
        return rf_model, iforest, scaler, features
    except Exception as e:
        raise RuntimeError(f"❌ Error loading artifacts: {e}")

def explain_prediction(label, confidence):
    if label == 'BENIGN':
        return f"""
✅ BENIGN TRAFFIC DETECTED (Confidence: {confidence*100:.2f}%)

This traffic is classified as safe and shows no signs of malicious activity.
Detailed characteristics:
- Normal frequency of requests and responses
- No unexpected port scans or flooding
- Communication patterns match known safe behaviors
- Consistent usage of legitimate services and protocols

While this traffic is safe, it's recommended to maintain continuous monitoring
to ensure threats are identified early and response times are fast.
"""
    else:
        return f"""
🚨 ANOMALOUS TRAFFIC DETECTED (Confidence: {confidence*100:.2f}%)

This traffic is highly suspicious and may indicate one or more of the following:
- Denial-of-Service (DoS) or Distributed DoS (DDoS) attacks
- Port scanning, reconnaissance, or lateral movement
- Attempts to exploit vulnerabilities or exfiltrate data
- Abnormally high traffic rates or malformed packet structures

Key anomaly indicators:
- Sharp deviations from learned benign patterns
- Detected by Isolation Forest based on statistical outliers
- Confirmed by Random Forest trained on real-world attack data

RECOMMENDED ACTIONS:
- Immediately isolate the source IP or subnet
- Check logs, firewall alerts, and IDS/IPS feedback
- Alert security teams and initiate incident response if needed
"""

def run_prediction(input_csv, model_dir="../models", output_txt=None):
    print(f"\n📂 Loading input CSV: {input_csv}")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ File not found: {input_csv}")
    
    df = pd.read_csv(input_csv)

    # Load model, scaler, and feature info
    rf_model, iforest, scaler, feature_columns = load_artifacts(model_dir)

    # Ensure compatibility with trained features
    missing_cols = [col for col in feature_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    df = df[feature_columns]

    # Preprocess
    X_scaled = scaler.transform(df)
    anomaly_scores = iforest.decision_function(X_scaled)

    # Predict
    start = time.time()
    predictions = rf_model.predict(X_scaled)
    probabilities = rf_model.predict_proba(X_scaled)
    end = time.time()

    total_time = end - start
    print(f"\n⏱️ Inference completed in {total_time:.2f} seconds.")

    # Compile prediction reports
    results = []
    for i, pred in enumerate(predictions):
        confidence = max(probabilities[i])
        explanation = explain_prediction(pred, confidence)
        results.append(f"🔹 Prediction: {pred}\n🔹 Confidence: {confidence*100:.2f}%\n{explanation}")

    # Print sample
    print("\n📌 Sample prediction:")
    print(results[0])

    # Save to output file if specified
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(entry + "\n" + "-"*80 + "\n")
        print(f"\n📝 Results saved to: {output_txt}")

    # Estimate accuracy if label column exists
    if 'Label' in df.columns:
        accuracy = accuracy_score(df['Label'], predictions)
        print(f"\n📊 Accuracy: {accuracy*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="🔍 Anomaly Detection Predictor")
    parser.add_argument("input_csv", help="Path to input CSV file inside the 'samples/' folder")
    parser.add_argument("--model_dir", default="../models", help="Directory where trained models are stored")
    parser.add_argument("--output_txt", help="Optional path to save detailed results")
    args = parser.parse_args()

    run_prediction(args.input_csv, args.model_dir, args.output_txt)

if __name__ == "__main__":
    main()
