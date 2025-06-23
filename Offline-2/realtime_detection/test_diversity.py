"""
Test the fixed real-time detection to verify diverse predictions.
"""

import requests
import time
import json

def test_multiple_predictions():
    """Test multiple predictions to see if we get variety."""
    
    base_url = "http://localhost:5001"
    
    print("Testing Multiple Predictions for Diversity")
    print("=" * 50)
    
    all_predictions = []
    all_confidences = []
    
    # Run 10 tests
    for test_num in range(1, 11):
        print(f"\nTest {test_num}/10:")
        
        # Start detection
        try:
            response = requests.post(f"{base_url}/start_detection")
            if response.status_code != 200:
                print(f"  ❌ Failed to start: {response.status_code}")
                continue
                
            # Wait for completion
            max_attempts = 20
            for attempt in range(max_attempts):
                response = requests.get(f"{base_url}/get_prediction")
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "unknown")
                    
                    if status == "complete":
                        website = result.get('website', 'Unknown')
                        confidence = result.get('confidence', 0)
                        all_predictions.append(website)
                        all_confidences.append(confidence)
                        
                        print(f"  ✅ Predicted: {website}")
                        print(f"     Confidence: {confidence:.3f}")
                        break
                    elif status == "error":
                        print(f"  ❌ Error: {result.get('error', 'Unknown')}")
                        break
                        
                time.sleep(1)
            else:
                print(f"  ⚠️  Timeout")
                
            # Small delay between tests
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    # Analyze results
    print(f"\n" + "=" * 50)
    print("RESULTS ANALYSIS")
    print("=" * 50)
    
    if all_predictions:
        from collections import Counter
        prediction_counts = Counter(all_predictions)
        
        print(f"Total successful predictions: {len(all_predictions)}")
        print(f"Prediction distribution:")
        for website, count in prediction_counts.items():
            percentage = (count / len(all_predictions)) * 100
            avg_conf = sum(c for p, c in zip(all_predictions, all_confidences) if p == website) / count
            print(f"  {website}: {count}/{len(all_predictions)} ({percentage:.1f}%) - avg confidence: {avg_conf:.3f}")
        
        # Check if we have diversity
        unique_predictions = len(prediction_counts)
        print(f"\nDiversity: {unique_predictions}/3 websites predicted")
        
        if unique_predictions == 1:
            print("❌ BIAS ISSUE: Only predicting one website")
        elif unique_predictions == 2:
            print("⚠️  PARTIAL BIAS: Only predicting two websites")
        else:
            print("✅ GOOD DIVERSITY: Predicting all websites")
    else:
        print("❌ No successful predictions to analyze")

if __name__ == "__main__":
    print("Make sure the Flask app is running...")
    time.sleep(2)
    test_multiple_predictions()
