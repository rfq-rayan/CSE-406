"""
Debug script to examine what data is actually being collected by the real-time app.
"""

import json
import torch
import numpy as np
from models import ComplexFingerprintClassifier

def analyze_collected_data():
    """Analyze the actual data collection process."""
    
    print("Real-Time Data Collection Analysis")
    print("=" * 50)
    
    # Load model
    model = ComplexFingerprintClassifier(1000, 128, 3)
    model.load_state_dict(torch.load('../code/saved_models/complex_model.pth'))
    model.eval()
    
    website_names = ['https://cse.buet.ac.bd/moodle/', 'https://google.com', 'https://prothomalo.com']
    
    # Simulate the real data collection process
    print("Simulating real-time data collection...")
    
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time
    
    try:
        # Setup Chrome exactly like in the real app
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        print("Collecting real timing data...")
        timing_data = []
        start_time = time.time()
        collection_duration = 5  # Shorter for testing
        
        while time.time() - start_time < collection_duration:
            try:
                driver.get("data:text/html,<html><body><script>console.log('timing');</script></body></html>")
                
                timing = driver.execute_script("""
                    return {
                        navigationStart: performance.timing.navigationStart,
                        loadEventEnd: performance.timing.loadEventEnd,
                        domContentLoaded: performance.timing.domContentLoadedEventEnd,
                        connectEnd: performance.timing.connectEnd,
                        responseEnd: performance.timing.responseEnd
                    };
                """)
                
                if timing['loadEventEnd'] > 0:
                    relative_time = timing['loadEventEnd'] - timing['navigationStart']
                    timing_data.append(relative_time % 100)
                
                time.sleep(0.01)
                
            except Exception as e:
                timing_data.append(np.random.randint(20, 80))
        
        driver.quit()
        
        print(f"Collected {len(timing_data)} timing points")
        
        # Pad to 1000 points like in the real app
        if len(timing_data) < 1000:
            while len(timing_data) < 1000:
                timing_data.extend(timing_data[:min(100, 1000 - len(timing_data))])
        timing_data = timing_data[:1000]
        
        print(f"Final timing data: {len(timing_data)} points")
        print(f"Sample values: {timing_data[:20]}")
        print(f"Mean: {np.mean(timing_data):.2f}")
        print(f"Std: {np.std(timing_data):.2f}")
        print(f"Range: [{min(timing_data)}, {max(timing_data)}]")
        
        # Make prediction
        trace_tensor = torch.FloatTensor(timing_data).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_website = website_names[predicted_idx]
        
        print(f"\nPrediction Results:")
        print(f"Predicted: {predicted_website}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities:")
        for i, (website, prob) in enumerate(zip(website_names, probabilities[0])):
            print(f"  {i}: {website} = {prob:.3f}")
            
        # Compare with real data from dataset
        print(f"\nComparison with dataset:")
        with open('../code/dataset.json', 'r') as f:
            dataset = json.load(f)
        
        complete_data = [entry for entry in dataset if len(entry['trace_data']) == 1000]
        
        for website in website_names:
            website_traces = [entry['trace_data'] for entry in complete_data if entry['website'] == website]
            if website_traces:
                sample_trace = website_traces[0]
                print(f"{website}:")
                print(f"  Dataset sample - Mean: {np.mean(sample_trace):.2f}, Std: {np.std(sample_trace):.2f}")
        
        print(f"Real-time data - Mean: {np.mean(timing_data):.2f}, Std: {np.std(timing_data):.2f}")
        
    except Exception as e:
        print(f"Error in real data collection: {e}")
        print("This explains why fallback is being used")

if __name__ == "__main__":
    analyze_collected_data()
