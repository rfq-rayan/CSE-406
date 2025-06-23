#!/usr/bin/env python3
"""
Analyze the training data distribution to understand model bias
"""

import json
import numpy as np
from collections import Counter

def analyze_training_data():
    """Analyze the distribution of websites and features in training data"""
    
    # Load the dataset
    dataset_path = "../code/dataset.json"
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Analyze website distribution
    print("\n" + "="*60)
    print("WEBSITE DISTRIBUTION IN TRAINING DATA")
    print("="*60)
    
    websites = [item['website'] for item in dataset]
    website_counts = Counter(websites)
    total_samples = len(websites)
    
    print(f"Total training samples: {total_samples}")
    print("\nDistribution:")
    for i, (website, count) in enumerate(website_counts.most_common()):
        percentage = (count / total_samples) * 100
        print(f"  {i}: {website}")
        print(f"     Count: {count} ({percentage:.1f}%)")
    
    # Analyze timing features
    print("\n" + "="*60)
    print("TIMING FEATURES ANALYSIS")
    print("="*60)
      all_features = []
    feature_stats_by_website = {}
    
    for item in dataset:
        website = item['website']
        features = item['trace_data']
        all_features.extend(features)
        
        if website not in feature_stats_by_website:
            feature_stats_by_website[website] = []
        feature_stats_by_website[website].extend(features)
    
    # Overall feature statistics
    features_array = np.array(all_features)
    print(f"Overall feature statistics:")
    print(f"  Mean: {np.mean(features_array):.2f}")
    print(f"  Std:  {np.std(features_array):.2f}")
    print(f"  Min:  {np.min(features_array):.2f}")
    print(f"  Max:  {np.max(features_array):.2f}")
    
    # Per-website feature statistics
    print(f"\nPer-website feature statistics:")
    for website, features in feature_stats_by_website.items():
        features_array = np.array(features)
        print(f"\n  {website}:")
        print(f"    Mean: {np.mean(features_array):.2f}")
        print(f"    Std:  {np.std(features_array):.2f}")
        print(f"    Min:  {np.min(features_array):.2f}")
        print(f"    Max:  {np.max(features_array):.2f}")
        print(f"    Samples: {len(features_array)}")
    
    # Feature range analysis
    print("\n" + "="*60)
    print("FEATURE RANGE ANALYSIS")
    print("="*60)
    
    # Check if features are in different ranges
    range_buckets = {
        "0-10": 0,
        "10-30": 0,
        "30-50": 0,
        "50-70": 0,
        "70+": 0
    }
    
    for feature in all_features:
        if feature < 10:
            range_buckets["0-10"] += 1
        elif feature < 30:
            range_buckets["10-30"] += 1
        elif feature < 50:
            range_buckets["30-50"] += 1
        elif feature < 70:
            range_buckets["50-70"] += 1
        else:
            range_buckets["70+"] += 1
    
    total_features = len(all_features)
    print("Feature value distribution:")
    for range_name, count in range_buckets.items():
        percentage = (count / total_features) * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_training_data()
