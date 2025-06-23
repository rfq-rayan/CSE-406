#!/usr/bin/env python3
"""
Test the model with actual training data patterns to see if the bias is in the model or data preprocessing
"""

import json
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from models import ComplexFingerprintClassifier

def test_model_with_actual_data():
    """Test the model with actual data from each website class"""
    
    # Configuration
    MODEL_PATH = "../code/saved_models/complex_model.pth"
    DATASET_PATH = "../code/dataset.json"
    INPUT_SIZE = 1000
    HIDDEN_SIZE = 128
    
    # Load dataset
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Filter to complete traces only
    complete_data = [entry for entry in data if len(entry['trace_data']) == INPUT_SIZE]
    websites = [entry['website'] for entry in complete_data]
    unique_websites = sorted(list(set(websites)))
    num_classes = len(unique_websites)
    
    print(f"Testing model with {num_classes} website classes:")
    for i, website in enumerate(unique_websites):
        print(f"  {i}: {website}")
    
    # Load model
    model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("\n" + "="*80)
    print("TESTING MODEL WITH ACTUAL TRAINING DATA")
    print("="*80)
    
    # Test with samples from each website
    predictions_summary = {}
    
    for target_website in unique_websites:
        print(f"\nTesting with samples from: {target_website}")
        print("-" * 50)
        
        # Get all samples for this website
        website_samples = [entry for entry in complete_data if entry['website'] == target_website]
        
        predictions = []
        confidences = []
        
        # Test with first 5 samples from this website
        for i, sample in enumerate(website_samples[:5]):
            trace_data = sample['trace_data']
            
            # Convert to tensor
            trace_tensor = torch.FloatTensor(trace_data).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(trace_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_website = unique_websites[predicted_class]
            predictions.append(predicted_website)
            confidences.append(confidence)
            
            # Show all class probabilities for first sample
            if i == 0:
                print(f"  Sample {i+1} - All class probabilities:")
                for j, website in enumerate(unique_websites):
                    prob = probabilities[0][j].item()
                    mark = " ★" if j == predicted_class else ""
                    print(f"    {website}: {prob:.4f}{mark}")
            
            print(f"  Sample {i+1}: Predicted {predicted_website} (confidence: {confidence:.3f}) {'✓' if predicted_website == target_website else '✗'}")
        
        # Summary for this website
        correct_predictions = sum(1 for p in predictions if p == target_website)
        avg_confidence = np.mean(confidences)
        
        predictions_summary[target_website] = {
            'correct': correct_predictions,
            'total': len(predictions),
            'accuracy': correct_predictions / len(predictions),
            'avg_confidence': avg_confidence,
            'predictions': predictions
        }
        
        print(f"  Summary: {correct_predictions}/{len(predictions)} correct ({correct_predictions/len(predictions)*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.3f}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_correct = sum(stats['correct'] for stats in predictions_summary.values())
    total_tested = sum(stats['total'] for stats in predictions_summary.values())
    
    print(f"Overall accuracy: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")
    
    for website, stats in predictions_summary.items():
        print(f"\n{website}:")
        print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
        print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
        print(f"  Predictions: {stats['predictions']}")
    
    # Check if there's a bias towards certain classes
    all_predictions = []
    for stats in predictions_summary.values():
        all_predictions.extend(stats['predictions'])
    
    print(f"\nPrediction distribution across all tests:")
    for website in unique_websites:
        count = all_predictions.count(website)
        percentage = count / len(all_predictions) * 100
        print(f"  {website}: {count}/{len(all_predictions)} ({percentage:.1f}%)")

if __name__ == "__main__":
    test_model_with_actual_data()
