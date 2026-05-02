"""
Master script to run all project modules sequentially
Run: python run_all.py
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    print("\n" + "="*60)
    print(f"🚀 {description}")
    print("="*60)
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    if result.returncode == 0:
        print(f"✅ Completed in {end_time - start_time:.2f} seconds")
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False

def main():
    print("="*60)
    print("🏦 BANK CHURN PREDICTION SYSTEM")
    print("Complete Project Execution")
    print("="*60)
    
    if not os.path.exists('src'):
        print("❌ Error: Please run this script from the project root directory")
        return
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('app', exist_ok=True)
    
    if not run_command("python src/01_data_preprocessing.py", "Step 1: Data Preprocessing"):
        return
    if not run_command("python src/02_feature_engineering.py", "Step 2: Feature Engineering"):
        return
    if not run_command("python src/03_model_training.py", "Step 3: Model Training"):
        return
    if not run_command("python src/04_model_evaluation.py", "Step 4: Model Evaluation"):
        return
    if not run_command("python src/05_shap_analysis.py", "Step 5: SHAP Analysis (Explainability)"):
        return
    
    print("\n" + "="*60)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   - models/churn_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/feature_importance.pkl")
    print("   - outputs/ (all visualizations and reports)")
    
    print("\n🚀 Next Steps:")
    print("   streamlit run app/streamlit_app.py")
    
    response = input("\n🎯 Launch Streamlit dashboard now? (y/n): ")
    if response.lower() == 'y':
        subprocess.run("streamlit run app/streamlit_app.py", shell=True)

if __name__ == "__main__":
    main()
