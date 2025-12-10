"""
Fix annotations script: Regenerate structured_annotations.json with real captions
from the original MSR_VTT.json file.
Handles FLAT frame structure: video0_frame_0.jpg, video0_frame_1.jpg, etc.
"""
import json
import os
from pathlib import Path
import re

def load_msrvtt_annotations(ann_path):
    """Load annotations from MSR_VTT.json"""
    print(f"Loading annotations from {ann_path}...")
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Found {len(data.get('annotations', []))} annotations")
    print(f"  Found {len(data.get('images', []))} videos")
    
    return data

def group_captions_by_video(annotations):
    """Group all captions by video_id"""
    video_captions = {}
    for ann in annotations:
        video_id = ann.get('image_id', '')
        caption = ann.get('caption', '')
        if video_id and caption:
            if video_id not in video_captions:
                video_captions[video_id] = []
            video_captions[video_id].append(caption)
    
    print(f"  Grouped captions for {len(video_captions)} videos")
    return video_captions

def get_frame_files_flat(frames_dir):
    """Get dictionary mapping video_id to list of frame files (flat structure)"""
    video_frames = {}
    
    # List all files in frames directory
    if not os.path.exists(frames_dir):
        return video_frames
    
    frame_pattern = re.compile(r'^(video\d+)_frame_(\d+)\.jpg$', re.IGNORECASE)
    
    for filename in os.listdir(frames_dir):
        match = frame_pattern.match(filename)
        if match:
            video_id = match.group(1)
            if video_id not in video_frames:
                video_frames[video_id] = []
            video_frames[video_id].append(filename)
    
    # Sort frames for each video
    for video_id in video_frames:
        video_frames[video_id].sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
    
    return video_frames

def create_fixed_annotations(msrvtt_root, output_file=None):
    """Create fixed structured annotations with real captions"""
    
    # Paths
    ann_path = os.path.join(msrvtt_root, 'annotation', 'MSR_VTT.json')
    frames_dir = os.path.join(msrvtt_root, 'frames')
    videos_dir = os.path.join(msrvtt_root, 'videos')
    
    if output_file is None:
        output_file = os.path.join(msrvtt_root, 'structured_annotations.json')
    
    # Backup old annotations
    if os.path.exists(output_file):
        backup_path = output_file.replace('.json', '_backup.json')
        print(f"Backing up old annotations to {backup_path}")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(output_file, backup_path)
    
    # Load MSR-VTT annotations
    msrvtt_data = load_msrvtt_annotations(ann_path)
    video_captions = group_captions_by_video(msrvtt_data.get('annotations', []))
    
    # Get frames using flat structure
    video_frames = get_frame_files_flat(frames_dir)
    
    print(f"\nFound frames for {len(video_frames)} videos")
    
    # Define train/test split based on MSR-VTT standard split
    # video0-video6512 = train, video6513-video7009 = val, video7010-video9999 = test
    def get_split(video_id):
        try:
            vid_num = int(video_id.replace('video', ''))
            if vid_num <= 6512:
                return 'train'
            elif vid_num <= 7009:
                return 'val'
            else:
                return 'test'
        except:
            return 'train'
    
    # Create structured annotations
    structured_annotations = []
    videos_with_captions = 0
    videos_without_captions = 0
    
    for video_id, frame_files in video_frames.items():
        if not frame_files:
            continue
        
        # Get captions for this video
        captions = video_captions.get(video_id, [])
        
        if captions:
            # Use the first caption as the primary caption
            # (MSR-VTT has ~20 captions per video, we pick one)
            caption = captions[0]
            videos_with_captions += 1
        else:
            # Fallback for videos without annotations
            caption = f"A video scene from {video_id}"
            videos_without_captions += 1
        
        # Determine split
        split = get_split(video_id)
        
        # Check for video file
        video_file = None
        for ext in ['.mp4', '.avi', '.webm']:
            vf = os.path.join(videos_dir, f"{video_id}{ext}")
            if os.path.exists(vf):
                video_file = f"{video_id}{ext}"
                break
        
        entry = {
            'video_id': video_id,
            'caption': caption,
            'all_captions': captions if captions else [caption],  # Store all captions for metrics
            'split': split,
            'video_file': video_file,
            'frame_files': frame_files,
            'has_frames': len(frame_files) > 0
        }
        
        structured_annotations.append(entry)
    
    # Sort by video_id
    structured_annotations.sort(key=lambda x: int(x['video_id'].replace('video', '')))
    
    # Save
    print(f"\nSaving {len(structured_annotations)} annotations to {output_file}")
    print(f"  Videos with real captions: {videos_with_captions}")
    print(f"  Videos without captions (fallback): {videos_without_captions}")
    
    # Count splits
    train_count = sum(1 for a in structured_annotations if a['split'] == 'train')
    val_count = sum(1 for a in structured_annotations if a['split'] == 'val')
    test_count = sum(1 for a in structured_annotations if a['split'] == 'test')
    print(f"  Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_annotations, f, indent=2, ensure_ascii=False)
    
    # Show sample
    print("\nSample annotations (first 3):")
    for ann in structured_annotations[:3]:
        cap_preview = ann['caption'][:70] if len(ann['caption']) > 70 else ann['caption']
        print(f"  {ann['video_id']}: {cap_preview}...")
    
    return structured_annotations

if __name__ == "__main__":
    # Auto-detect MSR-VTT root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    msrvtt_root = os.path.join(script_dir, 'dataset', 'MSRVTT')
    
    if not os.path.exists(msrvtt_root):
        print(f"Error: MSR-VTT dataset not found at {msrvtt_root}")
        exit(1)
    
    print("=" * 60)
    print("Fixing MSR-VTT Annotations (Flat Frame Structure)")
    print("=" * 60)
    
    create_fixed_annotations(msrvtt_root)
    
    print("\n" + "=" * 60)
    print("Done! Annotations have been corrected.")
    print("=" * 60)
