import os
import subprocess
import sys

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def train_model():
    print("📚 Starting training process...")
    try:
        subprocess.run([sys.executable, os.path.join("scripts", "train_model.py")], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running train_model.py: {e}")

def run_prediction():
    print("🔮 Running prediction...")
    try:
        subprocess.run([sys.executable, os.path.join("scripts", "predictor.py")], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running predictor.py: {e}")

def main():
    clear_console()
    print("🧠 Anomaly Detection CLI")
    print("=========================")
    print("1. Train Model")
    print("2. Run Prediction")
    print("0. Exit")

    choice = input("👉 Choose an option: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        run_prediction()
    elif choice == "0":
        print("👋 Exiting...")
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
