#!/usr/bin/env python3
"""
Compare synthetic data generation with actual training data patterns
"""

import json
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from models import ComplexFingerprintClassifier

def generate_synthetic_data_current_method():
    """Generate synthetic data using the current method in app.py"""
    
    # This is the current fallback method from app.py
    timing_data = []
    for i in range(1000):
        # Generate data in the same range as training data (mean ~40-42)
        base_value = 40 + 5 * np.sin(i * 0.1) + np.random.normal(0, 2)
        timing_value = max(20, min(70, base_value))
        timing_data.append(timing_value)
    
    return timing_data

def generate_realistic_synthetic_data(real_traces):
    """Generate more realistic synthetic data based on actual patterns"""
    
    # Pick a random real trace as a template
    template_trace = real_traces[np.random.randint(0, len(real_traces))]
    
    # Method 1: Add small noise to real trace
    synthetic_trace = []
    for value in template_trace:
        noise = np.random.normal(0, 0.3)  # Smaller noise
        new_value = value + noise
        synthetic_trace.append(new_value)
    
    return synthetic_trace

def test_synthetic_vs_real():
    """Test predictions on synthetic vs real data"""
    
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
    
    # Load model
    model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("Testing different data generation methods:")
    print("="*60)
    
    # Get all real traces for reference
    all_real_traces = [entry['trace_data'] for entry in complete_data]
    
    # Test 1: Current synthetic method
    print("\n1. Current Synthetic Method (sine wave + noise):")
    print("-" * 40)
    
    predictions_current = []
    for i in range(10):
        synthetic_data = generate_synthetic_data_current_method()
        
        # Analyze the data
        mean_val = np.mean(synthetic_data)
        std_val = np.std(synthetic_data)
        min_val = np.min(synthetic_data)
        max_val = np.max(synthetic_data)
        
        if i == 0:  # Show stats for first sample
            print(f"  Data stats: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        
        # Make prediction
        trace_tensor = torch.FloatTensor(synthetic_data).unsqueeze(0)
        with torch.no_grad():
            outputs = model(trace_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        predicted_website = unique_websites[predicted_class]
        predictions_current.append(predicted_website)
        
        if i < 3:  # Show first 3 predictions
            print(f"  Test {i+1}: {predicted_website} (confidence: {confidence:.3f})")
    
    # Summary for current method
    prediction_counts_current = {website: predictions_current.count(website) for website in unique_websites}
    print(f"  Summary: {prediction_counts_current}")
    
    # Test 2: Realistic synthetic method
    print("\n2. Realistic Synthetic Method (real data + small noise):")
    print("-" * 40)
    
    predictions_realistic = []
    for i in range(10):
        synthetic_data = generate_realistic_synthetic_data(all_real_traces)
        
        # Analyze the data
        mean_val = np.mean(synthetic_data)
        std_val = np.std(synthetic_data)
        min_val = np.min(synthetic_data)
        max_val = np.max(synthetic_data)
        
        if i == 0:  # Show stats for first sample
            print(f"  Data stats: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        
        # Make prediction
        trace_tensor = torch.FloatTensor(synthetic_data).unsqueeze(0)
        with torch.no_grad():
            outputs = model(trace_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        predicted_website = unique_websites[predicted_class]
        predictions_realistic.append(predicted_website)
        
        if i < 3:  # Show first 3 predictions
            print(f"  Test {i+1}: {predicted_website} (confidence: {confidence:.3f})")
    
    # Summary for realistic method
    prediction_counts_realistic = {website: predictions_realistic.count(website) for website in unique_websites}
    print(f"  Summary: {prediction_counts_realistic}")
    
    # Test 3: Real data baseline
    print("\n3. Real Data Baseline:")
    print("-" * 40)
    
    predictions_real = []
    for i in range(10):
        # Pick a random real trace
        real_sample = complete_data[np.random.randint(0, len(complete_data))]
        real_data = real_sample['trace_data']
        true_website = real_sample['website']
        
        # Make prediction
        trace_tensor = torch.FloatTensor(real_data).unsqueeze(0)
        with torch.no_grad():
            outputs = model(trace_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        predicted_website = unique_websites[predicted_class]
        predictions_real.append(predicted_website)
        
        if i < 3:  # Show first 3 predictions
            correct = "✓" if predicted_website == true_website else "✗"
            print(f"  Test {i+1}: {predicted_website} (true: {true_website}) {correct} (confidence: {confidence:.3f})")
    
    # Summary for real method
    prediction_counts_real = {website: predictions_real.count(website) for website in unique_websites}
    print(f"  Summary: {prediction_counts_real}")
    
    # Compare all methods
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Current Synthetic:   {prediction_counts_current}")
    print(f"Realistic Synthetic: {prediction_counts_realistic}")
    print(f"Real Data Baseline:  {prediction_counts_real}")

if __name__ == "__main__":
    test_synthetic_vs_real()
