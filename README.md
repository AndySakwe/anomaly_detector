# 🔍 Anomaly Detector CLI Tool

The **Anomaly Detector CLI Tool** is a command-line Python-based solution for detecting anomalies and Distributed Denial-of-Service (DDoS) attacks in network traffic using machine learning models like **Random Forest** and **Isolation Forest**. This project is tailored to efficiently analyze network behavior and determine whether a connection is **BENIGN** or an **ANOMALY**, and even classify the **type of attack** in grouped multiclass mode.

This tool is designed for **researchers**, **security analysts**, **students**, and **developers** who want to quickly analyze CSV files containing network logs and determine threats in a **portable**, **lightweight**, and **explainable** way.

---

## 🔧 Features

- ✅ Binary Anomaly Detection (`BENIGN` vs `ANOMALY`)
- ✅ Grouped Multiclass Detection (e.g., `DrDoS`, `UDP`, `TFTP`)
- ✅ Random Forest for classification
- ✅ Isolation Forest for anomaly scoring
- ✅ CLI-based prediction from any `.csv` file
- ✅ Explainability via per-row confidence scores and attack reasoning
- ✅ Auto-preprocessing: handles extra/missing columns
- ✅ Memory-efficient dataset sampling during training
- ✅ Compatible with Linux and Windows
- ✅ GitHub-ready with CLI installer (`setup.py`)
- ✅ Training and prediction logs with timestamps and inference times
- ✅ Exportable confusion matrix

---

## 📁 Project Structure
Anomaly detection/
├── anomaly_detector/
│ ├── init.py
│ ├── train_rfmodel.py # Training script
│ └── predictor.py # CLI prediction script
├── models/ # Trained model files (.joblib, .pkl)
├── data/ # Dataset CSV files (ignored from Git)
├── samples/ # One benign and one anomaly sample saved during training
├── results/ # Confusion matrix, reports, predictions
├── requirements.txt # Python dependencies
├── .gitignore # Git exclusions (models, data, etc.)
├── README.md # This file
└── setup.py # Installable CLI tool config


---

## 📦 Installation

1. **Clone the repository**:

git clone https://github.com/AndySakwe/anomaly_detector.git
cd anomaly_detector

2. Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows

3.Install dependencies:

pip install -r requirements.txt

🚀 Usage
1.🧠 Train the model

Train a hybrid Random Forest + Isolation Forest model on labeled data:
python main.py --mode train
This will:
Load .csv files from data/
Preprocess and sample memory-efficiently
Train the model

Save:
models/rf_model.joblib
models/scaler.pkl
models/if_model.joblib
models/feature_columns.pkl
Generate a confusion matrix and save it to results/

2.📈 Predict from CSV
Predict anomalies from any new file:

python main.py --mode predict --input path/to/your.csv
Output includes:
Predicted class (BENIGN, ANOMALY, UDP, etc.)
Confidence score (0-100%)
Severity
Explanation (e.g., "Likely UDP flood: High packet rate, low variance")
CSV of predictions saved to results/predictions.csv
Plaintext report saved to results/report.txt

3.🧠 Supported Attack Classes
Grouped into classes to reduce noise:
DrDoS: DNS, LDAP, MSSQL, NTP, etc.
UDP: UDP attacks including lag and flooding
TFTP: Trivial File Transfer Protocol
BENIGN: Normal traffic
ANOMALY: Any unknown or suspicious behavior

🧪 Dataset Instructions
The tool is compatible with CIC-DDoS2019 and any CSV file with numerical features and a Label or GroupLabel column.

Important:
Do not upload raw datasets to GitHub (use .gitignore)
Place them inside data/
Each file should be clean and standardized (handled by script)

🛠 Install as CLI Tool
Convert the project to a Linux-wide CLI tool:
pip install .
Then use anywhere:
anomaly-detector --mode predict --input your.csv

🤝 Contribution Guide
If you'd like to contribute:

Fork the repository

Create a new branch (feature/your-feature)

Commit your changes

Push to your fork

Open a Pull Request

All contributions are welcome: bug fixes, improvements, attack class enhancements, deep learning models, etc.

📜 License
This project is licensed under the MIT License. You are free to modify, distribute, and use it for personal or commercial purposes.

🙏 Acknowledgments
CIC-DDoS2019 Dataset by Canadian Institute for Cybersecurity

GitHub for version control and collaboration

The User — for taking cybersecurity seriously

💬 Support
If you encounter issues or have questions:
Create an issue on GitHub
Or email uwuaandrew@gmail.com
This project is a practical and scalable solution for anomaly detection and can be integrated into larger cybersecurity pipelines or used for academic research.
