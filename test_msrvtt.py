
import os
import json

def test_msrvtt_structure(msrvtt_path):
    """Test the MSR VTT dataset structure"""
    
    print(f"Testing MSR VTT structure at: {msrvtt_path}")
    
    # Check for annotation file
    ann_file = os.path.join(msrvtt_path, 'annotation', 'MSR_VTT.json')
    if not os.path.exists(ann_file):
        print(f"ERROR: MSR_VTT.json not found at {ann_file}")
        return
    
    print(f"Found annotation file: {ann_file}")
    
    # Load and inspect
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nFile type: {type(data)}")
    
    if isinstance(data, dict):
        print("Top-level keys:")
        for key in data.keys():
            print(f"  {key}: {type(data[key])}")
            
            if isinstance(data[key], list):
                print(f"    Length: {len(data[key])}")
                if len(data[key]) > 0:
                    print(f"    First item type: {type(data[key][0])}")
                    if isinstance(data[key][0], dict):
                        print(f"    First item keys: {list(data[key][0].keys())[:10]}")
                        
                        # Show a sample if it has video_id and caption
                        sample = data[key][0]
                        if 'video_id' in sample and 'caption' in sample:
                            print(f"    Sample video_id: {sample['video_id']}")
                            print(f"    Sample caption: {sample['caption'][:100]}...")
    
    # Try to find annotations in different formats
    annotations = []
    
    # Format 1: Direct lists
    if isinstance(data, dict):
        for key in ['train', 'validate', 'test', 'val', 'training']:
            if key in data and isinstance(data[key], list):
                print(f"\nFound '{key}' list with {len(data[key])} items")
                for i, item in enumerate(data[key][:3]):
                    if isinstance(item, dict):
                        print(f"  Item {i+1}:")
                        for k, v in item.items():
                            if k in ['video_id', 'id', 'caption', 'sentence']:
                                print(f"    {k}: {v}")
                        
                        if 'video_id' in item and 'caption' in item:
                            annotations.append({
                                'video_id': str(item['video_id']),
                                'caption': item['caption']
                            })
    
    print(f"\nTotal annotations found: {len(annotations)}")
    
    # Save sample annotations
    if annotations:
        sample_file = os.path.join(msrvtt_path, 'sample_annotations.json')
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(annotations[:100], f, indent=2)
        print(f"\nSaved sample annotations to: {sample_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter MSR VTT dataset path: ").strip()
    
    test_msrvtt_structure(path)
    