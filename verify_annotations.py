"""Verify the fixed annotations"""
import json

with open('dataset/MSRVTT/structured_annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total annotations: {len(data)}")

print("\nFirst 5 samples:")
for a in data[:5]:
    caption = a['caption'][:70]
    print(f"  {a['video_id']} ({a['split']}): {caption}...")

print("\nSplit counts:")
train = sum(1 for a in data if a['split'] == 'train')
val = sum(1 for a in data if a['split'] == 'val')
test = sum(1 for a in data if a['split'] == 'test')
print(f"  Train: {train}, Val: {val}, Test: {test}")

# Check if captions look real
print("\nCaption quality check:")
placeholder_count = sum(1 for a in data if 'A video showing content' in a['caption'])
real_count = len(data) - placeholder_count
print(f"  Real captions: {real_count}")
print(f"  Placeholder captions: {placeholder_count}")
