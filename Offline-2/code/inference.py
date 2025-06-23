"""
Inference script for website fingerprinting classification using trained models.
This script demonstrates how to load and use the trained models for prediction.
"""

import json
import torch
import numpy as np
from train import FingerprintClassifier, ComplexFingerprintClassifier

# Configuration
MODEL_PATH = "saved_models/complex_model.pth"  # Use the best performing model
DATASET_PATH = "dataset.json"
INPUT_SIZE = 1000
HIDDEN_SIZE = 128

def load_model_and_metadata():
    """Load the trained model and dataset metadata."""
    
    # Load dataset to get website names and number of classes
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Filter to complete traces only
    complete_data = [entry for entry in data if len(entry['trace_data']) == INPUT_SIZE]
    websites = [entry['website'] for entry in complete_data]
    unique_websites = sorted(list(set(websites)))
    num_classes = len(unique_websites)
    
    print(f"Loaded metadata: {num_classes} classes")
    for i, website in enumerate(unique_websites):
        print(f"  {i}: {website}")
    
    # Initialize and load model
    model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}")
    
    return model, unique_websites

def predict_website(model, trace_data, website_names):
    """Predict the website for a given trace."""
    
    # Convert to tensor
    trace_tensor = torch.FloatTensor(trace_data).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(trace_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    predicted_website = website_names[predicted_class]
    
    return predicted_website, confidence, probabilities[0].tolist()

def demo_inference():
    """Demonstrate inference on some examples from the dataset."""
    
    print("Website Fingerprinting Inference Demo")
    print("=" * 50)
    
    # Load model and metadata
    model, website_names = load_model_and_metadata()
    
    # Load some test examples
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Filter to complete traces
    complete_data = [entry for entry in data if len(entry['trace_data']) == INPUT_SIZE]
    
    print(f"\nRunning inference on {min(5, len(complete_data))} example traces:")
    print("-" * 50)
    
    for i, entry in enumerate(complete_data[:5]):
        trace = entry['trace_data']
        true_website = entry['website']
        
        # Make prediction
        predicted_website, confidence, all_probs = predict_website(model, trace, website_names)
        
        print(f"\nExample {i+1}:")
        print(f"  True website: {true_website}")
        print(f"  Predicted website: {predicted_website}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Correct: {'✓' if predicted_website == true_website else '✗'}")
        
        # Show probabilities for all classes
        print(f"  All probabilities:")
        for j, (website, prob) in enumerate(zip(website_names, all_probs)):
            print(f"    {j}: {website[:50]:<50} {prob:.3f}")

if __name__ == "__main__":
    demo_inference()
