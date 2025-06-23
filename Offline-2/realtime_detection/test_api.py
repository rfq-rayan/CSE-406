"""
Test script to verify the real-time detection works after the ChromeDriver fix.
"""

import requests
import time
import json

def test_realtime_detection():
    """Test the real-time detection endpoint."""
    
    base_url = "http://localhost:5001"
    
    print("Testing Real-Time Website Detection")
    print("=" * 40)
    
    # Test 1: Start detection
    print("\n1. Starting detection...")
    try:
        response = requests.post(f"{base_url}/start_detection")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Detection started: {result}")
        else:
            print(f"   ❌ Failed to start: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return
    
    # Test 2: Monitor progress
    print("\n2. Monitoring progress...")
    max_attempts = 30  # 30 seconds max
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/get_prediction")
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                print(f"   Attempt {attempt+1}: Status = {status}")
                
                if status == "complete":
                    print(f"   ✅ Detection completed!")
                    print(f"   Predicted website: {result.get('website', 'Unknown')}")
                    print(f"   Confidence: {result.get('confidence', 0):.3f}")
                    if "note" in result:
                        print(f"   Note: {result['note']}")
                    break
                elif status == "error":
                    print(f"   ❌ Detection failed: {result.get('error', 'Unknown error')}")
                    break
                
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error checking status: {e}")
            break
    else:
        print("   ⚠️  Detection timed out")
    
    # Test 3: Test demo mode
    print("\n3. Testing demo mode...")
    try:
        response = requests.post(f"{base_url}/demo_predict")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Demo prediction successful!")
            print(f"   Predicted: {result.get('predicted_website', 'Unknown')}")
            print(f"   Actual: {result.get('true_website', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Correct: {result.get('correct', False)}")
        else:
            print(f"   ❌ Demo failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Demo error: {e}")

if __name__ == "__main__":
    print("Make sure the Flask app is running on localhost:5001")
    print("Press Enter to start testing...")
    input()
    
    test_realtime_detection()
    
    print("\n" + "=" * 40)
    print("Testing complete!")
    print("The Win32 error should now be resolved with fallback handling.")
