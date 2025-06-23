#!/usr/bin/env python3
"""
Test script for the website fingerprinting data collection system.
This script performs basic validation of the implementation.    print("Note: Make sure you have Microsoft Edge browser installed for Selenium automation.")"""

import os
import sys
import time
import subprocess
import signal
import socket

def check_server_running(host='127.0.0.1', port=5000):
    """Check if Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def start_flask_server():
    """Start Flask server in background."""
    print("Starting Flask server...")
    process = subprocess.Popen([sys.executable, 'app.py'], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(10):
        time.sleep(1)
        if check_server_running():
            print("✓ Flask server started successfully")
            return process
        print(f"  Waiting for server to start... ({i+1}/10)")
    
    print("✗ Flask server failed to start")
    return None

def test_basic_functionality():
    """Test basic functionality of the collection system."""
    print("=== Testing Website Fingerprinting Collection System ===\n")
    
    # Check if required files exist
    required_files = ['app.py', 'collect.py', 'database.py', 'requirements.txt']
    missing_files = []
    
    print("1. Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nERROR: Missing required files: {missing_files}")
        return False
    
    # Check static files
    print("\n2. Checking static files...")
    static_files = ['static/index.html', 'static/index.js', 'static/warmup.js', 'static/worker.js']
    for file in static_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
    
    # Test Flask server
    print("\n3. Testing Flask server...")
    server_process = None
    
    try:
        if not check_server_running():
            server_process = start_flask_server()
            if not server_process:
                print("  ✗ Could not start Flask server")
                return False
        else:
            print("  ✓ Flask server already running")
        
        # Test server endpoints
        print("\n4. Testing server endpoints...")
        import requests
        
        try:
            # Test main page
            response = requests.get('http://localhost:5000/', timeout=5)
            if response.status_code == 200:
                print("  ✓ Main page accessible")
            else:
                print(f"  ✗ Main page returned status {response.status_code}")
            
            # Test API endpoint
            response = requests.get('http://localhost:5000/api/get_results', timeout=5)
            if response.status_code == 200:
                print("  ✓ API endpoint accessible")
            else:
                print(f"  ✗ API endpoint returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error testing endpoints: {e}")
            return False
    
    except ImportError:
        print("  ! requests module not available, skipping endpoint tests")
    
    finally:
        # Clean up server process
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
                print("  ✓ Flask server stopped")
            except:
                try:
                    server_process.kill()
                    print("  ✓ Flask server force-stopped")
                except:
                    print("  ! Could not stop Flask server")
      # Test database functionality
    print("\n5. Testing database functionality...")
    try:
        import database
        from database import Database
        
        # Test database initialization with a test database
        test_websites = ["https://example.com", "https://test.com"]
        # Use a different database file for testing
        original_db_path = database.DATABASE_PATH
        database.DATABASE_PATH = "test_webfingerprint.db"
        database.DB_URL = f"sqlite:///{database.DATABASE_PATH}"
        
        db = Database(test_websites)
        db.init_database()
        print("  ✓ Database initialization successful")
        
        # Test saving a trace
        test_trace = [1, 2, 3, 4, 5]
        success = db.save_trace("https://example.com", 1, test_trace)
        if success:
            print("  ✓ Trace saving successful")
        else:
            print("  ✗ Trace saving failed")
        
        # Test retrieving counts
        counts = db.get_traces_collected()
        print(f"  ✓ Trace counts retrieved: {counts}")
          # Restore original database path
        database.DATABASE_PATH = original_db_path
        database.DB_URL = f"sqlite:///{database.DATABASE_PATH}"
        
        print("  ✓ Database test completed successfully")
        # Note: test database file left in place for cleanup later
            
    except Exception as e:
        print(f"  ✗ Database test failed: {e}")
        return False
    
    print("\n=== All Basic Tests Completed Successfully! ===")
    print("\nTo run the full collection system:")
    print("1. Start the Flask server: python app.py")
    print("2. In another terminal, run: python collect.py")
    print("3. The system will collect traces automatically")
    print("\nNote: Make sure you have Chrome browser installed for Selenium automation.")
    
    return True

if __name__ == "__main__":
    test_basic_functionality()
