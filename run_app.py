#!/usr/bin/env python3
"""
Script to run the Streamlit application
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🏥 Medical Data Analysis Dashboard")
    print("=" * 40)
    
    # Check if CSV file exists
    if not os.path.exists("_Thesis - Sheet1.csv"):
        print("❌ CSV file not found!")
        print("Please ensure '_Thesis - Sheet1.csv' is in the current directory")
        sys.exit(1)
    
    # Install requirements
    print("📦 Installing requirements...")
    if install_requirements():
        print("🚀 Starting Streamlit application...")
        run_streamlit()
    else:
        print("❌ Failed to start application due to installation errors")