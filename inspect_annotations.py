
import os
import json

def test_msrvtt_annotations(msrvtt_path):
    """Test loading MSR VTT annotations"""
    
    print(f"Testing MSR VTT annotations at: {msrvtt_path}")
    
    # Check annotation file
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
        print("Dictionary keys:")
        for key in data.keys():
            print(f"  {key}: {type(data[key])}")
            
            if isinstance(data[key], list):
                print(f"    Length: {len(data[key])}")
                if len(data[key]) > 0:
                    item = data[key][0]
                    print(f"    First item type: {type(item)}")
                    if isinstance(item, dict):
                        print(f"    First item keys: {list(item.keys())[:10]}")
                        
                        # Show values for important keys
                        for k in ['video_id', 'id', 'caption', 'sentence']:
                            if k in item:
                                print(f"      {k}: {item[k]}")
    
    # Try to extract annotations
    annotations = []
    
    if isinstance(data, dict):
        # Try common MSR VTT structures
        possible_keys = ['train', 'validate', 'test', 'val', 'training', 'annotations', 'sentences']
        
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                print(f"\nFound list under key '{key}' with {len(data[key])} items")
                
                # Take first few items
                for i, item in enumerate(data[key][:3]):
                    if isinstance(item, dict):
                        print(f"  Item {i+1}:")
                        # Print important fields
                        for k, v in item.items():
                            if k in ['video_id', 'id', 'caption', 'sentence']:
                                print(f"    {k}: {v}")
                        
                        # Extract if it has video_id and caption
                        vid = item.get('video_id', item.get('id', ''))
                        cap = item.get('caption', item.get('sentence', ''))
                        
                        if vid and cap:
                            annotations.append({
                                'video_id': str(vid),
                                'caption': cap
                            })
    
    print(f"\nTotal annotations extracted: {len(annotations)}")
    
    # Save sample annotations
    if annotations:
        sample_file = os.path.join(msrvtt_path, 'extracted_annotations_sample.json')
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(annotations[:50], f, indent=2)
        print(f"\nSaved sample annotations to: {sample_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter MSR VTT dataset path: ").strip()
    
    test_msrvtt_annotations(path)
    