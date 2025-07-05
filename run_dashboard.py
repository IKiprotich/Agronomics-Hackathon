#!/usr/bin/env python3
"""
Agricultural Dashboard Runner
Enhanced version with correlation analysis, Sankey diagrams, and effectiveness categorization.
"""

import subprocess
import sys
import os
import time

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Please manually install: pip install -r requirements.txt")
        return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Agricultural Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("🔄 To restart the server, press Ctrl+C and run this script again")
    print("📊 Features included:")
    print("  - County-Crop/Livestock Correlation Analysis")
    print("  - Sankey Flow Diagrams")
    print("  - Effectiveness Categorization (Highest/Moderate/Least Effective)")
    print("  - Interactive Data Explorer")
    print("  - Enhanced ML Models")
    print("  - Tabulated Results with Clear Legends")
    print("-" * 50)
    
    try:
        # Run streamlit with proper configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "agricultural_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main function"""
    print("🌾 Agricultural Data Analytics Dashboard")
    print("=" * 50)
    
    # Check if dashboard file exists
    if not os.path.exists("agricultural_dashboard.py"):
        print("❌ agricultural_dashboard.py not found in current directory")
        print("Please ensure you're in the correct directory")
        return
    
    # Check if requirements file exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        print("Please ensure requirements.txt is in the current directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Small delay to ensure packages are properly installed
    time.sleep(2)
    
    # Run dashboard
    while True:
        success = run_dashboard()
        if success:
            restart = input("\n🔄 Restart dashboard? (y/n): ").lower().strip()
            if restart != 'y':
                break
        else:
            print("❌ Dashboard failed to start")
            break
    
    print("👋 Thanks for using the Agricultural Dashboard!")

if __name__ == "__main__":
    main() 