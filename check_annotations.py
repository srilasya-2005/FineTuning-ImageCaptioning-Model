"""Script to inspect the MSR_VTT.json annotation structure and evaluate model checkpoints."""
import json
import os

def inspect_msrvtt_annotations():
    """Inspect the structure of MSR_VTT.json annotations."""
    ann_path = "dataset/MSRVTT/annotation/MSR_VTT.json"
    
    print("=== Inspecting MSR_VTT.json ===")
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    print(f"Keys: {list(data.keys())}")
    print(f"Number of images (videos): {len(data.get('images', []))}")
    print(f"Number of annotations: {len(data.get('annotations', []))}")
    
    anns = data.get('annotations', [])
    if anns:
        print("\nFirst 5 annotations:")
        for a in anns[:5]:
            caption = a.get('caption', 'N/A')
            video_id = a.get('image_id', 'N/A')
            print(f"  Video: {video_id}, Caption: {caption[:80]}...")
    
    # Check how many captions per video
    video_caption_counts = {}
    for a in anns:
        vid = a.get('image_id', 'unknown')
        video_caption_counts[vid] = video_caption_counts.get(vid, 0) + 1
    
    if video_caption_counts:
        sample_videos = list(video_caption_counts.items())[:5]
        print(f"\nCaptions per video (sample):")
        for vid, count in sample_videos:
            print(f"  {vid}: {count} captions")
    
    return data

def inspect_structured_annotations():
    """Inspect what was used for training."""
    struct_path = "dataset/MSRVTT/structured_annotations.json"
    
    print("\n=== Inspecting structured_annotations.json ===")
    with open(struct_path, 'r') as f:
        data = json.load(f)
    
    print(f"Number of entries: {len(data)}")
    if data:
        print("\nFirst 3 entries:")
        for entry in data[:3]:
            print(f"  Video: {entry.get('video_id')}")
            print(f"  Caption: {entry.get('caption')}")
            print(f"  Split: {entry.get('split')}")
            print()

def check_checkpoints():
    """List and check checkpoint sizes."""
    ckpt_dir = "msrvtt_fine_tuned_captioning/checkpoints"
    
    print("\n=== Checking Checkpoints ===")
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    print(f"Found {len(ckpts)} checkpoints")
    
    # Show last 5 checkpoint sizes
    print("\nLast 10 checkpoints (with sizes):")
    for ckpt in ckpts[-10:]:
        path = os.path.join(ckpt_dir, ckpt)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        status = "OK" if size_mb > 900 else "CORRUPTED!"
        print(f"  {ckpt}: {size_mb:.1f} MB - {status}")

if __name__ == "__main__":
    inspect_msrvtt_annotations()
    inspect_structured_annotations()
    check_checkpoints()
