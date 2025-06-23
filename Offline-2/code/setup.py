#!/usr/bin/env python3
"""
Setup script for the Website Fingerprinting Data Collection System.
This script helps install dependencies and verify the system setup.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages."""
    print("Installing required Python packages...")
    
    # First try to upgrade pip and setuptools
    try:
        print("  Upgrading pip and setuptools...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools"])
        print("  ✓ pip and setuptools upgraded")
    except subprocess.CalledProcessError as e:
        print(f"  ! Warning: Could not upgrade pip/setuptools: {e}")
    
    # Define packages to install individually
    packages = [
        "flask==2.3.3",
        "matplotlib>=3.8.0",  # Use newer version compatible with Python 3.12
        "numpy>=1.25.0",      # Use newer version compatible with Python 3.12
        "selenium==4.15.0",
        "webdriver-manager==4.0.1",
        "sqlalchemy>=2.0.0",
        "requests"  # For testing
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✓ {package}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n✗ Failed to install packages: {failed_packages}")
        print("Please try installing them manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    else:
        print("✓ All packages installed successfully")
        return True

def check_chrome_installation():
    """Check if Chrome browser is installed."""
    print("Checking Chrome browser installation...")
    
    # Common Chrome paths on Windows
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
    ]
    
    chrome_found = False
    for path in chrome_paths:
        if os.path.exists(path):
            print(f"  ✓ Chrome found at: {path}")
            chrome_found = True
            break
    
    if not chrome_found:
        print("  ✗ Chrome browser not found")
        print("  Please install Google Chrome from: https://www.google.com/chrome/")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating necessary directories...")
    
    directories = [
        "static/heatmaps"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✓ Created directory: {directory}")
        else:
            print(f"  ✓ Directory exists: {directory}")
    
    return True

def main():
    """Main setup function."""
    print("=== Website Fingerprinting Data Collection Setup ===\n")
    
    success = True
    
    # Step 1: Install Python packages
    if not install_requirements():
        success = False
    
    print()
    
    # Step 2: Check Chrome installation
    if not check_chrome_installation():
        success = False
    
    print()
    
    # Step 3: Create directories
    if not create_directories():
        success = False
    
    print()
    
    if success:
        print("=== Setup Completed Successfully! ===")
        print("\nNext steps:")
        print("1. Run the test script: python test_system.py")
        print("2. Start the Flask server: python app.py")
        print("3. Run the collection script: python collect.py")
        print("\nFor testing, you can also:")
        print("- Open http://localhost:5000 in your browser")
        print("- Manually test the fingerprinting interface")
    else:
        print("=== Setup Failed ===")
        print("Please fix the issues above before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
