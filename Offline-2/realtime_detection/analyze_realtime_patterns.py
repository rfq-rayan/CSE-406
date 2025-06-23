#!/usr/bin/env python3
"""
Analyze the real-time data collection patterns to understand the bias
"""

import requests
import time
import json
import numpy as np

def analyze_realtime_patterns():
    """Collect multiple real-time samples and analyze their patterns"""
    
    base_url = "http://localhost:5001"
    
    print("Analyzing Real-Time Data Collection Patterns")
    print("=" * 60)
    
    patterns = []
    
    for i in range(5):
        print(f"\nSample {i+1}/5:")
        print("-" * 30)
        
        # Start detection
        try:
            response = requests.post(f"{base_url}/start_detection")
            if response.status_code != 200:
                print(f"  ❌ Failed to start detection: {response.status_code}")
                continue
        except Exception as e:
            print(f"  ❌ Connection error: {e}")
            continue
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{base_url}/get_prediction")
                data = response.json()
                
                status = data.get('status', 'unknown')
                print(f"  Status: {status}")
                
                if status == 'complete':
                    website = data.get('website', 'Unknown')
                    confidence = data.get('confidence', 0.0)
                    print(f"  ✅ Predicted: {website} (confidence: {confidence:.3f})")
                    
                    # Try to get the actual timing data for analysis
                    # (This would require modifying the app to expose the raw data)
                    patterns.append({
                        'website': website,
                        'confidence': confidence,
                        'sample_id': i+1
                    })
                    break
                elif status == 'error':
                    error_msg = data.get('error', 'Unknown error')
                    print(f"  ❌ Error: {error_msg}")
                    break
                else:
                    time.sleep(1)  # Wait a bit more
                    
            except Exception as e:
                print(f"  ❌ Polling error: {e}")
                break
        else:
            print(f"  ❌ Timeout waiting for prediction")
        
        # Small delay between samples
        time.sleep(2)
    
    # Analyze the patterns
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    if not patterns:
        print("❌ No successful samples collected")
        return
    
    websites = [p['website'] for p in patterns]
    confidences = [p['confidence'] for p in patterns]
    
    print(f"Total samples: {len(patterns)}")
    print(f"Unique predictions: {len(set(websites))}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    
    print(f"\nPrediction distribution:")
    unique_websites = list(set(websites))
    for website in unique_websites:
        count = websites.count(website)
        avg_conf = np.mean([p['confidence'] for p in patterns if p['website'] == website])
        print(f"  {website}: {count}/{len(patterns)} ({count/len(patterns)*100:.1f}%) - avg conf: {avg_conf:.3f}")
    
    if len(set(websites)) == 1:
        print("\n❌ BIAS DETECTED: All predictions are the same!")
        print("   Real-time data collection is producing identical patterns.")
    elif len(set(websites)) == len(patterns):
        print("\n✅ GOOD DIVERSITY: All predictions are different!")
    else:
        print(f"\n⚠️  PARTIAL DIVERSITY: {len(set(websites))}/{len(patterns)} unique predictions")

if __name__ == "__main__":
    analyze_realtime_patterns()
