import json

with open('dataset.json', 'r') as f:
    data = json.load(f)

print('Total entries:', len(data))
print('First entry keys:', list(data[0].keys()))
print('First trace length:', len(data[0]['trace_data']))
print('First few trace values:', data[0]['trace_data'][:10])
print()
print('Checking all trace lengths...')
for i, entry in enumerate(data[:5]):
    print(f'Entry {i}: website={entry["website"]}, trace_length={len(entry["trace_data"])}')

print()
print('Checking for inconsistent trace lengths...')
trace_lengths = [len(entry["trace_data"]) for entry in data]
unique_lengths = set(trace_lengths)
print('Unique trace lengths found:', unique_lengths)

if len(unique_lengths) > 1:
    print('PROBLEM: Inconsistent trace lengths!')
    for length in unique_lengths:
        count = trace_lengths.count(length)
        print(f'  Length {length}: {count} traces')
