import json

print("Loading dataset...")
with open('dataset.json', 'r') as f:
    data = json.load(f)

print(f"Total traces: {len(data)}")

websites = {}
for entry in data:
    website = entry['website']
    websites[website] = websites.get(website, 0) + 1

print("\nTraces per website:")
for site, count in websites.items():
    print(f"  {site}: {count}")

if data:
    print(f"\nSample trace length: {len(data[0]['trace_data'])}")
    print(f"Sample trace data (first 10 values): {data[0]['trace_data'][:10]}")
