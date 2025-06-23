#!/usr/bin/env python3
"""
Check which websites have complete traces (1000 data points)
"""

import json

def check_complete_traces():
    dataset_path = "../code/dataset.json"
    INPUT_SIZE = 1000
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total dataset entries: {len(data)}")
    
    # Filter to complete traces only
    complete_data = [entry for entry in data if len(entry['trace_data']) == INPUT_SIZE]
    incomplete_data = [entry for entry in data if len(entry['trace_data']) != INPUT_SIZE]
    
    print(f"Complete traces (exactly {INPUT_SIZE} points): {len(complete_data)}")
    print(f"Incomplete traces: {len(incomplete_data)}")
    
    # Check websites in complete data
    websites_complete = [entry['website'] for entry in complete_data]
    unique_websites_complete = sorted(list(set(websites_complete)))
    
    print(f"\nWebsites with complete traces:")
    for i, website in enumerate(unique_websites_complete):
        count = websites_complete.count(website)
        print(f"  {i}: {website} ({count} samples)")
    
    # Check websites in incomplete data
    if incomplete_data:
        websites_incomplete = [entry['website'] for entry in incomplete_data]
        unique_websites_incomplete = sorted(list(set(websites_incomplete)))
        
        print(f"\nWebsites with incomplete traces:")
        for website in unique_websites_incomplete:
            count = websites_incomplete.count(website)
            lengths = [len(entry['trace_data']) for entry in incomplete_data if entry['website'] == website]
            print(f"  {website} ({count} samples, lengths: {set(lengths)})")

if __name__ == "__main__":
    check_complete_traces()
