"""
Setup script for Real-Time Website Detection app.
This script checks dependencies and prepares the environment.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_packages():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_model_files():
    """Check if trained model files exist."""
    model_path = "../code/saved_models/complex_model.pth"
    dataset_path = "../code/dataset.json"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please run the training script first: python ../code/train.py")
        return False
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("   Please run data collection first")
        return False
    
    print("✅ Model and dataset files found")
    return True

def check_chrome():
    """Check if Chrome is available."""
    try:
        import selenium
        from webdriver_manager.chrome import ChromeDriverManager
        
        # Try to get ChromeDriver
        ChromeDriverManager().install()
        print("✅ Chrome WebDriver available")
        return True
    except Exception as e:
        print(f"❌ Chrome WebDriver issue: {e}")
        return False

def main():
    """Main setup function."""
    print("Real-Time Website Detection Setup")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Packages", install_packages),
        ("Model Files", check_model_files),
        ("Chrome WebDriver", check_chrome),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ Setup completed successfully!")
        print("\nTo run the real-time detection app:")
        print("  python app.py")
        print("\nThen open your browser to: http://localhost:5001")
    else:
        print("❌ Setup failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
