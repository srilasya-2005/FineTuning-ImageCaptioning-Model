"""Verify the fixed annotations have real captions."""
import json

ann_path = 'dataset/MSRVTT/structured_annotations.json'
with open(ann_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total annotations: {len(data)}")

if data:
    print(f"\nSample caption: {data[0]['caption'][:100]}...")
    
    train = sum(1 for x in data if x['split'] == 'train')
    val = sum(1 for x in data if x['split'] == 'val')
    test = sum(1 for x in data if x['split'] == 'test')
    print(f"\nTrain: {train}, Val: {val}, Test: {test}")
    
    # Check if these are real captions (not placeholders)
    placeholders = sum(1 for x in data if 'video scene' in x['caption'].lower())
    print(f"\nPlaceholder captions: {placeholders}/{len(data)}")
    
    if placeholders == 0 or placeholders < len(data) * 0.1:
        print("\n✓ Annotations have real MSR-VTT captions!")
    else:
        print("\n✗ WARNING: Many placeholder captions detected!")
else:
    print("ERROR: No annotations found!")
