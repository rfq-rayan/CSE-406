"""
Test script to analyze model bias and prediction patterns.
"""

import json
import torch
import numpy as np
from collections import Counter
import sys
import os

# Add the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

def load_model_and_data():
    """Load the trained model and dataset."""
    
    # Import model locally to avoid path issues
    from models import ComplexFingerprintClassifier
    
    # Load dataset
    with open('../code/dataset.json', 'r') as f:
        data = json.load(f)
    
    # Filter complete traces
    complete_data = [entry for entry in data if len(entry['trace_data']) == 1000]
    
    # Get website names
    websites = [entry['website'] for entry in complete_data]
    unique_websites = sorted(list(set(websites)))
    
    # Load model
    model = ComplexFingerprintClassifier(1000, 128, len(unique_websites))
    model.load_state_dict(torch.load('../code/saved_models/complex_model.pth'))
    model.eval()
    
    return model, complete_data, unique_websites

def test_model_predictions():
    """Test the model on all traces in the dataset."""
    
    model, complete_data, unique_websites = load_model_and_data()
    
    print("Model Bias Analysis")
    print("=" * 50)
    print(f"Testing model on {len(complete_data)} traces from {len(unique_websites)} websites")
    print()
    
    # Test on all traces
    predictions = []
    true_labels = []
    confidences = []
    
    for entry in complete_data:
        trace = entry['trace_data']
        true_website = entry['website']
        
        # Make prediction
        trace_tensor = torch.FloatTensor(trace).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_website = unique_websites[predicted_idx]
        
        predictions.append(predicted_website)
        true_labels.append(true_website)
        confidences.append(confidence)
    
    # Analyze results
    print("Prediction Distribution:")
    prediction_counts = Counter(predictions)
    for website, count in prediction_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"  {website}: {count}/{len(predictions)} ({percentage:.1f}%)")
    
    print(f"\nTrue Label Distribution:")
    true_counts = Counter(true_labels)
    for website, count in true_counts.items():
        percentage = (count / len(true_labels)) * 100
        print(f"  {website}: {count}/{len(true_labels)} ({percentage:.1f}%)")
    
    # Accuracy per class
    print(f"\nAccuracy by Class:")
    for website in unique_websites:
        true_positive = sum(1 for t, p in zip(true_labels, predictions) 
                           if t == website and p == website)
        total_true = true_labels.count(website)
        accuracy = (true_positive / total_true) * 100 if total_true > 0 else 0
        print(f"  {website}: {true_positive}/{total_true} ({accuracy:.1f}%)")
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    overall_accuracy = (correct / len(predictions)) * 100
    print(f"\nOverall Accuracy: {correct}/{len(predictions)} ({overall_accuracy:.1f}%)")
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    for website in unique_websites:
        website_confidences = [c for p, c in zip(predictions, confidences) if p == website]
        if website_confidences:
            avg_conf = np.mean(website_confidences)
            print(f"  {website}: Average confidence = {avg_conf:.3f}")

def test_synthetic_patterns():
    """Test the synthetic data generation patterns."""
    
    print("\n" + "=" * 50)
    print("Synthetic Data Analysis")
    print("=" * 50)
    
    # Test the updated patterns used in the real-time app
    base_patterns = {
        0: {"mean": 41.54, "std": 0.93, "name": "BUET Moodle"},      # BUET Moodle pattern
        1: {"mean": 41.44, "std": 0.93, "name": "Google"},          # Google pattern
        2: {"mean": 39.79, "std": 2.19, "name": "Prothom Alo"}     # Prothom Alo pattern
    }
    
    model, _, unique_websites = load_model_and_data()
    
    print("Testing improved synthetic patterns...")
    
    for pattern_idx, pattern_info in base_patterns.items():
        print(f"\nPattern {pattern_idx} (intended for {pattern_info['name']}):")
        
        # Generate multiple synthetic traces with this pattern
        predictions = []
        confidences = []
        
        for test_run in range(10):  # Test 10 times
            # Generate synthetic data using the new method
            timing_data = []
            for i in range(1000):
                # Use actual statistical characteristics from real data
                base_value = np.random.normal(pattern_info["mean"], pattern_info["std"])
                
                # Add positional variation similar to real data
                if i % 50 < 5:  # Some positions have different characteristics
                    base_value *= 0.7
                
                timing_value = max(10, min(90, base_value))
                timing_data.append(timing_value)
            
            # Make prediction
            trace_tensor = torch.FloatTensor(timing_data).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(trace_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            predicted_website = unique_websites[predicted_idx]
            predictions.append(predicted_website)
            confidences.append(confidence)
        
        # Analyze results for this pattern
        prediction_counts = Counter(predictions)
        print(f"  Predictions over 10 runs:")
        for website, count in prediction_counts.items():
            avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == website])
            print(f"    {website}: {count}/10 times (avg confidence: {avg_conf:.3f})")
        
        # Calculate accuracy for this pattern
        intended_website = unique_websites[pattern_idx]
        correct_predictions = predictions.count(intended_website)
        accuracy = (correct_predictions / len(predictions)) * 100
        print(f"  Accuracy: {correct_predictions}/10 ({accuracy:.1f}%)")

if __name__ == "__main__":
    test_model_predictions()
    test_synthetic_patterns()
