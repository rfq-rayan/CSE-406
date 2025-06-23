"""
Test script to diagnose ChromeDriver issues and verify the fix.
"""

import os
import sys
import subprocess

def test_chromedriver_approaches():
    """Test different ChromeDriver initialization approaches."""
    
    print("Testing ChromeDriver Approaches")
    print("=" * 40)
    
    # Test 1: Check if Chrome is installed
    print("\n1. Checking Chrome installation...")
    try:
        import winreg
        
        # Check Chrome registry entries
        chrome_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
        ]
        
        chrome_found = False
        for path in chrome_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                chrome_exe = winreg.QueryValue(key, None)
                print(f"   ✅ Chrome found: {chrome_exe}")
                chrome_found = True
                winreg.CloseKey(key)
                break
            except FileNotFoundError:
                continue
        
        if not chrome_found:
            print("   ❌ Chrome not found in registry")
    except Exception as e:
        print(f"   ⚠️  Could not check Chrome: {e}")
    
    # Test 2: Check Python architecture
    print("\n2. Checking Python architecture...")
    import platform
    print(f"   Python version: {sys.version}")
    print(f"   Architecture: {platform.architecture()}")
    print(f"   Platform: {platform.platform()}")
    
    # Test 3: Try different ChromeDriver approaches
    approaches = [
        ("Local ChromeDriver", test_local_chromedriver),
        ("ChromeDriverManager", test_chromedriver_manager),
        ("System PATH", test_system_chromedriver)
    ]
    
    for name, test_func in approaches:
        print(f"\n3. Testing {name}...")
        try:
            test_func()
            print(f"   ✅ {name} works!")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")

def test_local_chromedriver():
    """Test local ChromeDriver from main project."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    
    service = Service("../code/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("data:text/html,<html><body>Test</body></html>")
    driver.quit()

def test_chromedriver_manager():
    """Test ChromeDriverManager."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("data:text/html,<html><body>Test</body></html>")
    driver.quit()

def test_system_chromedriver():
    """Test system ChromeDriver from PATH."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=options)
    driver.get("data:text/html,<html><body>Test</body></html>")
    driver.quit()

def test_fallback_demo():
    """Test the fallback synthetic data generation."""
    print("\n4. Testing fallback synthetic data generation...")
    
    try:
        import numpy as np
        
        # Generate synthetic timing data
        timing_data = []
        base_patterns = {
            0: [45, 35, 42, 43, 44],
            1: [55, 48, 52, 50, 53],
            2: [38, 33, 40, 37, 41]
        }
        
        pattern_choice = np.random.choice(list(base_patterns.keys()))
        base_pattern = base_patterns[pattern_choice]
        
        for i in range(1000):  # INPUT_SIZE
            base_value = base_pattern[i % len(base_pattern)]
            noise = np.random.normal(0, 2)
            timing_value = max(10, min(90, base_value + noise))
            timing_data.append(timing_value)
        
        print(f"   ✅ Generated {len(timing_data)} synthetic timing points")
        print(f"   Sample values: {timing_data[:10]}")
        
    except Exception as e:
        print(f"   ❌ Fallback failed: {e}")

if __name__ == "__main__":
    test_chromedriver_approaches()
    test_fallback_demo()
    
    print("\n" + "=" * 40)
    print("Diagnosis complete!")
    print("\nIf ChromeDriver tests fail, the app will automatically")
    print("fall back to synthetic data generation for demonstration.")
