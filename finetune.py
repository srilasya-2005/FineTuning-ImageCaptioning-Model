import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import Dataset as HFDataset
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import cv2
import gc
import random
import traceback
import glob
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class MSRVTTDatasetProcessor:
    """Process MSR VTT dataset from the given folder structure"""
    
    @staticmethod
    def extract_frames_from_videos(video_dir, frames_dir, num_frames_per_video=3):
        """Extract frames from videos and save them"""
        os.makedirs(frames_dir, exist_ok=True)
        
        # Filter out macOS resource fork files (._ files)
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and not f.startswith('._')]
        print(f"Found {len(video_files)} valid video files (excluding resource fork files)")
        
        for video_file in tqdm(video_files, desc="Extracting frames"):
            video_path = os.path.join(video_dir, video_file)
            video_id = video_file.replace('.mp4', '')
            
            try:
                # Open video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    print(f"Warning: {video_file} has 0 frames, skipping...")
                    cap.release()
                    continue
                
                # Calculate frame indices to sample
                if total_frames < num_frames_per_video:
                    frame_indices = list(range(total_frames))
                else:
                    step = max(1, total_frames // num_frames_per_video)
                    frame_indices = [i * step for i in range(min(num_frames_per_video, total_frames))]
                
                # Extract and save frames
                for idx, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Save frame
                        frame_filename = f"{video_id}_frame_{idx}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    else:
                        print(f"  Warning: Could not read frame {frame_idx} from {video_file}")
                
                cap.release()
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
    
    @staticmethod
    def inspect_annotation_file(ann_file):
        """Inspect the structure of an annotation file"""
        print(f"\nInspecting annotation file: {ann_file}")
        
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"File structure type: {type(data)}")
            
            if isinstance(data, dict):
                print("Keys in dictionary:")
                for key in data.keys():
                    if isinstance(data[key], list):
                        print(f"  {key}: list with {len(data[key])} items")
                        if len(data[key]) > 0:
                            print(f"    First item type: {type(data[key][0])}")
                            if isinstance(data[key][0], dict):
                                print(f"    Keys in first item: {list(data[key][0].keys())[:5]}...")
                    elif isinstance(data[key], dict):
                        print(f"  {key}: dict with {len(data[key])} keys")
                    else:
                        print(f"  {key}: {type(data[key])}")
                
                # Show sample if 'annotations' key exists
                if 'annotations' in data and isinstance(data['annotations'], list) and len(data['annotations']) > 0:
                    print("\nSample annotation (first item):")
                    print(json.dumps(data['annotations'][0], indent=2))
                
                # Show sample if 'sentences' key exists  
                if 'sentences' in data and isinstance(data['sentences'], list) and len(data['sentences']) > 0:
                    print("\nSample sentence (first item):")
                    print(json.dumps(data['sentences'][0], indent=2))
                    
            elif isinstance(data, list):
                print(f"List with {len(data)} items")
                if len(data) > 0:
                    print(f"First item type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"Keys in first item: {list(data[0].keys())}")
                        print("\nSample item (first):")
                        print(json.dumps(data[0], indent=2))
                        
        except Exception as e:
            print(f"Error inspecting file: {e}")
    
    @staticmethod
    def load_annotations(annotation_dir):
        """Load annotations from the MSR VTT annotation files"""
        
        print(f"Looking for annotation files in: {annotation_dir}")
        
        # Check common annotation file locations
        possible_locations = [
            os.path.join(annotation_dir, 'annotation'),
            os.path.join(annotation_dir, 'annotations'),
            os.path.join(annotation_dir, 'train_val_annotation'),
            annotation_dir,  # Check root directory
        ]
        
        annotation_files = []
        
        for location in possible_locations:
            if os.path.exists(location):
                print(f"Checking location: {location}")
                # Look for JSON files
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if file.endswith('.json') and not file.startswith('._'):  # Skip resource fork files
                            full_path = os.path.join(root, file)
                            annotation_files.append(full_path)
                            print(f"  Found: {file}")
        
        if not annotation_files:
            print(f"No annotation files found in {annotation_dir}")
            print("Available files/directories in dataset root:")
            for item in os.listdir(annotation_dir):
                print(f"  {item}")
            return []
        
        print(f"\nFound {len(annotation_files)} annotation files")
        
        # Inspect each annotation file first
        for ann_file in annotation_files:
            MSRVTTDatasetProcessor.inspect_annotation_file(ann_file)
        
        annotations = []
        
        for ann_file in annotation_files:
            try:
                print(f"\nLoading annotation file: {os.path.basename(ann_file)}")
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different annotation formats
                if isinstance(data, dict):
                    # Check for MSR VTT format
                    if 'annotations' in data and isinstance(data['annotations'], list):
                        print(f"  Format: MSR VTT with 'annotations' key")
                        for ann in data['annotations']:
                            if isinstance(ann, dict):
                                # Extract video_id and caption
                                video_id = str(ann.get('video_id', ''))
                                caption = ann.get('caption', ann.get('sentence', ''))
                                
                                if video_id and caption:
                                    annotations.append({
                                        'video_id': video_id,
                                        'caption': caption,
                                        'split': ann.get('split', 'train'),
                                    })
                    
                    # Check for alternative format with 'sentences' and 'videos'
                    elif 'sentences' in data and 'videos' in data:
                        print(f"  Format: MSR VTT with 'sentences' and 'videos' keys")
                        
                        # Create video id to info mapping
                        video_info = {}
                        for video in data['videos']:
                            video_id = str(video.get('video_id', ''))
                            video_info[video_id] = video
                        
                        for sentence in data['sentences']:
                            video_id = str(sentence.get('video_id', ''))
                            caption = sentence.get('caption', '')
                            
                            if video_id and caption:
                                annotations.append({
                                    'video_id': video_id,
                                    'caption': caption,
                                    'split': video_info.get(video_id, {}).get('split', 'train'),
                                })
                
                elif isinstance(data, list):
                    print(f"  Format: List with {len(data)} items")
                    for item in data:
                        if isinstance(item, dict):
                            video_id = str(item.get('video_id', item.get('id', '')))
                            caption = item.get('caption', item.get('sentence', ''))
                            
                            if video_id and caption:
                                annotations.append({
                                    'video_id': video_id,
                                    'caption': caption,
                                    'split': item.get('split', 'train'),
                                })
                
                print(f"  Loaded {len(annotations)} total annotations so far")
                
            except json.JSONDecodeError as e:
                print(f"  JSON decode error in {ann_file}: {e}")
            except Exception as e:
                print(f"  Error loading {ann_file}: {e}")
                traceback.print_exc()
        
        print(f"\nTotal annotations loaded: {len(annotations)}")
        
        # Show some sample annotations
        if annotations:
            print("\nSample annotations:")
            for i in range(min(5, len(annotations))):
                print(f"  {i+1}. Video ID: {annotations[i]['video_id']}")
                print(f"     Caption: {annotations[i]['caption'][:100]}...")
                print(f"     Split: {annotations[i].get('split', 'N/A')}")
        
        return annotations
    
    @staticmethod
    def create_basic_annotations(data_dir):
        """Create basic annotations from video files when no annotation files are found"""
        video_dir = os.path.join(data_dir, 'videos', 'all')
        annotations = []
        
        if os.path.exists(video_dir):
            # Filter out macOS resource fork files
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and not f.startswith('._')]
            
            for video_file in video_files:
                video_id = video_file.replace('.mp4', '')
                
                # Clean video_id by removing 'video' prefix if present
                clean_id = video_id.replace('video', '')
                
                # Try to convert to int, if fails use original
                try:
                    vid_num = int(clean_id)
                    split = 'train' if vid_num < 7000 else 'test'
                except ValueError:
                    split = 'train'  # Default to train
                
                # Create a simple placeholder caption
                annotations.append({
                    'video_id': video_id,
                    'caption': f'A video showing content from {video_id}',
                    'split': split,
                })
        
        print(f"Created {len(annotations)} basic annotations from video files")
        return annotations
    
    @staticmethod
    def create_structured_annotations(msrvtt_root, annotations, output_file='structured_annotations.json'):
        """Create structured annotations with video frame information"""
        
        if not annotations:
            print("No annotations to structure")
            return []
        
        # Get video files (excluding resource fork files)
        video_dir = os.path.join(msrvtt_root, 'videos', 'all')
        video_files = {}
        
        if os.path.exists(video_dir):
            for f in os.listdir(video_dir):
                if f.endswith('.mp4') and not f.startswith('._'):
                    video_id = f.replace('.mp4', '')
                    video_files[video_id] = f
        
        # Get frame files
        frames_dir = os.path.join(msrvtt_root, 'frames')
        frame_files = {}
        
        if os.path.exists(frames_dir):
            for f in os.listdir(frames_dir):
                if f.endswith('.jpg') and not f.startswith('._'):
                    # Extract video_id from frame filename
                    if '_frame_' in f:
                        video_id = f.split('_frame_')[0]
                    else:
                        video_id = f.replace('.jpg', '')
                    
                    if video_id not in frame_files:
                        frame_files[video_id] = []
                    frame_files[video_id].append(f)
        
        print(f"\nVideo files found: {len(video_files)}")
        print(f"Frames found for {len(frame_files)} videos")
        
        # Create structured annotations
        structured_ann = []
        
        for ann in annotations:
            video_id = str(ann['video_id'])
            
            # Check if video exists or frames exist
            if video_id in video_files or video_id in frame_files:
                structured_ann.append({
                    **ann,
                    'video_file': video_files.get(video_id, ''),
                    'frame_files': frame_files.get(video_id, []),
                    'has_frames': video_id in frame_files
                })
            else:
                # Try to find similar video IDs
                possible_video_ids = [vid for vid in video_files.keys() if video_id in vid or vid in video_id]
                if possible_video_ids:
                    # Use the first matching video ID
                    matched_id = possible_video_ids[0]
                    structured_ann.append({
                        **ann,
                        'video_file': video_files.get(matched_id, ''),
                        'frame_files': frame_files.get(matched_id, []),
                        'has_frames': matched_id in frame_files,
                        'original_video_id': video_id,
                        'matched_video_id': matched_id
                    })
        
        # Save structured annotations
        output_path = os.path.join(msrvtt_root, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_ann, f, indent=2)
        
        print(f"Saved {len(structured_ann)} structured annotations to {output_path}")
        
        # Show statistics
        if structured_ann:
            with_frames = sum(1 for ann in structured_ann if ann.get('has_frames', False))
            print(f"  Annotations with frames: {with_frames}/{len(structured_ann)}")
            
            # Show samples
            print("\nSample structured annotations:")
            for i in range(min(3, len(structured_ann))):
                ann = structured_ann[i]
                print(f"  {i+1}. Video: {ann['video_id']}")
                print(f"     Caption: {ann['caption'][:80]}...")
                print(f"     Has frames: {ann.get('has_frames', False)}")
                if ann.get('frame_files'):
                    print(f"     Frame count: {len(ann['frame_files'])}")
        
        return structured_ann

class MSRVTTCaptioningDataset(Dataset):
    """Dataset for MSR VTT image captioning"""
    
    def __init__(self, 
                 msrvtt_root,
                 annotations,
                 processor,
                 tokenizer,
                 max_length=128,
                 image_size=224,
                 frames_per_video=1,
                 use_augmentation=False):
        
        self.msrvtt_root = msrvtt_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.use_augmentation = use_augmentation
        
        # Filter out annotations for resource fork files
        self.annotations = []
        frames_dir = os.path.join(msrvtt_root, 'frames')
        
        for ann in annotations:
            video_id = str(ann['video_id'])
            
            # Skip if video_id starts with ._ (resource fork)
            if video_id.startswith('._'):
                continue
                
            # Check if frames exist for this video
            frame_pattern = os.path.join(frames_dir, f"{video_id}_frame_*.jpg")
            frame_matches = glob.glob(frame_pattern)
            
            if frame_matches:
                self.annotations.append(ann)
            else:
                # Also check for video_id.jpg (single frame)
                single_frame = os.path.join(frames_dir, f"{video_id}.jpg")
                if os.path.exists(single_frame):
                    self.annotations.append(ann)
        
        print(f"Filtered to {len(self.annotations)} annotations with available frames")
        
        if not self.annotations:
            print("Warning: No annotations with frames found!")
            if os.path.exists(frames_dir):
                frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg') and not f.startswith('._')])
                print(f"Total frames in directory (excluding ._ files): {frame_count}")
        
        # Create frame cache - store paths only, not loaded images
        self.frame_cache = {}
        if self.annotations:
            self.load_frames_to_cache()
    
    def load_frames_to_cache(self):
        """Preload frame information to cache"""
        print("Loading frame information to cache...")
        
        frames_dir = os.path.join(self.msrvtt_root, 'frames')
        
        for ann in tqdm(self.annotations, desc="Caching frames"):
            video_id = str(ann['video_id'])
            
            # Look for frames in frames directory (excluding resource fork files)
            frame_pattern = os.path.join(frames_dir, f"{video_id}_frame_*.jpg")
            frame_files = [f for f in glob.glob(frame_pattern) if not os.path.basename(f).startswith('._')]
            
            # Also check for video_id.jpg (single frame)
            single_frame = os.path.join(frames_dir, f"{video_id}.jpg")
            if os.path.exists(single_frame) and not os.path.basename(single_frame).startswith('._'):
                frame_files.append(single_frame)
            
            if frame_files:
                self.frame_cache[video_id] = frame_files
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        try:
            ann = self.annotations[idx]
            video_id = str(ann['video_id'])
            
            # Get frame paths
            frame_paths = self.frame_cache.get(video_id, [])
            
            if not frame_paths:
                # Try to find frames on the fly
                frames_dir = os.path.join(self.msrvtt_root, 'frames')
                frame_pattern = os.path.join(frames_dir, f"{video_id}_frame_*.jpg")
                frame_paths = [f for f in glob.glob(frame_pattern) if not os.path.basename(f).startswith('._')]
                
                # Also check for video_id.jpg
                single_frame = os.path.join(frames_dir, f"{video_id}.jpg")
                if os.path.exists(single_frame) and not os.path.basename(single_frame).startswith('._'):
                    frame_paths.append(single_frame)
            
            if not frame_paths:
                # Return a placeholder with black image
                pixel_values = torch.zeros(3, self.image_size, self.image_size)
                caption = ann.get('caption', '')
                
                labels = self.tokenizer(
                    caption,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze()
                
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    'pixel_values': pixel_values,
                    'labels': labels,
                    'video_id': video_id,
                    'caption': caption,
                    'split': ann.get('split', 'train')
                }
            
            # Select frames (randomly or sequentially)
            if len(frame_paths) > self.frames_per_video:
                selected_frames = random.sample(frame_paths, self.frames_per_video)
            else:
                selected_frames = frame_paths
            
            # Load and process the first frame
            frame_path = selected_frames[0]
            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert('RGB')
                
                # Apply augmentation if enabled
                if self.use_augmentation and random.random() > 0.5:
                    image = self.apply_augmentation(image)
                
                # Process image
                pixel_values = self.processor(
                    images=image,
                    return_tensors="pt",
                    size={"height": self.image_size, "width": self.image_size}
                ).pixel_values.squeeze()
            else:
                pixel_values = torch.zeros(3, self.image_size, self.image_size)
            
            # Get caption
            caption = ann.get('caption', '')
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            # Tokenize caption
            labels = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            # Replace padding token id with -100 for loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'pixel_values': pixel_values,
                'labels': labels,
                'video_id': video_id,
                'caption': caption,
                'split': ann.get('split', 'train')
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            return self.create_dummy_sample()
    
    def create_dummy_sample(self):
        """Create a dummy sample for error cases"""
        pixel_values = torch.zeros(3, self.image_size, self.image_size)
        
        labels = self.tokenizer(
            "A video frame",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'video_id': 'dummy',
            'caption': 'A video frame',
            'split': 'train'
        }
    
    def apply_augmentation(self, image):
        """Apply data augmentation to image"""
        try:
            from torchvision import transforms
            
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            ])
            
            return aug_transform(image)
        except:
            return image

class MSRVTTFineTuner:
    def __init__(self,
                 msrvtt_root,
                 model_name="nlpconnect/vit-gpt2-image-captioning",
                 output_dir="./msrvtt_fine_tuned_model",
                 device=None):
        
        self.msrvtt_root = msrvtt_root
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize processor and tokenizer
        print(f"Loading processor and tokenizer from {model_name}...")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for video captioning
        special_tokens_dict = {
            'additional_special_tokens': [
                '<video>', '</video>',
                '<scene>', '</scene>',
                '<action>', '</action>',
                '<person>', '</person>',
                '<object>', '</object>'
            ]
        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens for video captioning")
        
        # Load model
        print(f"Loading model from {model_name}...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # CRITICAL FIX: Set decoder_start_token_id for GPT-2 decoder
        # For GPT-2, we need to set decoder_start_token_id to tokenizer.bos_token_id or tokenizer.cls_token_id
        # Since GPT-2 doesn't have a bos_token, we'll use tokenizer.cls_token_id if it exists,
        # otherwise we'll use tokenizer.eos_token_id
        
        if self.tokenizer.cls_token_id is not None:
            decoder_start_token_id = self.tokenizer.cls_token_id
        elif self.tokenizer.bos_token_id is not None:
            decoder_start_token_id = self.tokenizer.bos_token_id
        else:
            # GPT-2 doesn't have bos_token, so we use eos_token_id
            decoder_start_token_id = self.tokenizer.eos_token_id
        
        print(f"Setting decoder_start_token_id to: {decoder_start_token_id}")
        print(f"Tokenizer bos_token_id: {self.tokenizer.bos_token_id}")
        print(f"Tokenizer cls_token_id: {self.tokenizer.cls_token_id}")
        print(f"Tokenizer eos_token_id: {self.tokenizer.eos_token_id}")
        
        # Set the configuration
        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Set forced_bos_token_id if it exists
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.model.config.forced_bos_token_id = self.tokenizer.bos_token_id
        
        # Set generation parameters for video captioning
        self.model.config.max_length = 128
        self.model.config.num_beams = 4
        self.model.config.temperature = 0.7
        self.model.config.early_stopping = True
        
        # Resize token embeddings
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to device
        self.model.to(self.device)
        
        print("Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Verify configuration
        print(f"\nModel configuration:")
        print(f"  decoder_start_token_id: {self.model.config.decoder_start_token_id}")
        print(f"  pad_token_id: {self.model.config.pad_token_id}")
        print(f"  bos_token_id: {self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 'N/A'}")
        print(f"  cls_token_id: {self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else 'N/A'}")
    
    def prepare_dataset(self, annotations, split='train', frames_per_video=1, use_augmentation=False):
        """Prepare dataset for specific split"""
        
        # Filter annotations by split
        if split == 'train':
            split_annotations = [ann for ann in annotations 
                               if ann.get('split') in ['train', 'train_val', 'train-val', 'training']]
        elif split == 'val':
            split_annotations = [ann for ann in annotations 
                               if ann.get('split') in ['val', 'validate', 'validation', 'valid']]
        elif split == 'test':
            split_annotations = [ann for ann in annotations 
                               if ann.get('split') in ['test', 'testing']]
        else:
            split_annotations = annotations
        
        print(f"Preparing {split} dataset with {len(split_annotations)} samples")
        
        # If no split annotations, use all
        if not split_annotations and split != 'test':
            print(f"No {split} annotations found, using all annotations")
            split_annotations = annotations
        
        # Create dataset
        dataset = MSRVTTCaptioningDataset(
            msrvtt_root=self.msrvtt_root,
            annotations=split_annotations,
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_length=128,
            image_size=224,
            frames_per_video=frames_per_video,
            use_augmentation=use_augmentation and split == 'train'
        )
        
        return dataset
    
    def train_with_custom_loop(self,
                               train_dataset,
                               val_dataset=None,
                               num_epochs=3,
                               batch_size=2,
                               learning_rate=3e-5,
                               warmup_steps=50,
                               logging_steps=10,
                               eval_steps=20,
                               save_steps=50,
                               gradient_accumulation_steps=2,
                               fp16=False,
                               use_wandb=False):
        """Train the model using a custom training loop"""
        
        if use_wandb:
            try:
                wandb.init(
                    project="msrvtt-captioning-finetuning",
                    config={
                        "model": self.model_name,
                        "dataset": "MSR-VTT",
                        "epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate
                    }
                )
            except:
                print("Warning: WandB initialization failed, continuing without it")
                use_wandb = False
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None
        
        # Training loop
        print("\nStarting training with custom loop...")
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0
            
            train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch_idx, batch in enumerate(train_progress):
                try:
                    # Move batch to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass (with mixed precision if enabled)
                    if fp16 and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                pixel_values=pixel_values,
                                labels=labels,
                                return_dict=True
                            )
                            loss = outputs.loss
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = self.model(
                            pixel_values=pixel_values,
                            labels=labels,
                            return_dict=True
                        )
                        loss = outputs.loss
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                    
                    scheduler.step()
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / (batch_idx + 1)
                    
                    train_progress.set_postfix({'loss': avg_loss})
                    
                    # Logging
                    global_step += 1
                    if use_wandb and global_step % logging_steps == 0:
                        wandb.log({
                            'train_loss': loss.item(),
                            'learning_rate': scheduler.get_last_lr()[0],
                            'step': global_step,
                            'epoch': epoch
                        })
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(epoch, global_step, avg_loss)
                    
                    # Evaluation
                    if val_loader and global_step % eval_steps == 0:
                        val_loss = self.evaluate(val_loader, fp16)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_model(self.output_dir)
                            print(f"New best model saved with val_loss: {val_loss:.4f}")
                        
                        if use_wandb:
                            wandb.log({
                                'val_loss': val_loss,
                                'step': global_step,
                                'epoch': epoch
                            })
                    
                except Exception as e:
                    print(f"\nError in training batch {batch_idx}: {e}")
                    traceback.print_exc()
                    
                    # Skip this batch and continue
                    continue
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, global_step, avg_epoch_loss, is_epoch_end=True)
        
        print("\nTraining completed!")
        
        # Save final model
        self.save_model(self.output_dir)
        
        if use_wandb:
            wandb.finish()
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }
    
    def evaluate(self, val_loader, fp16=False):
        """Evaluate the model on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if fp16 and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                pixel_values=pixel_values,
                                labels=labels,
                                return_dict=True
                            )
                            loss = outputs.loss
                    else:
                        outputs = self.model(
                            pixel_values=pixel_values,
                            labels=labels,
                            return_dict=True
                        )
                        loss = outputs.loss
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        print(f"Validation loss: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, epoch, step, loss, is_epoch_end=False, max_checkpoints=2):
        """Save a checkpoint and clean up old ones to save storage"""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_epoch_end:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pt')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{step}_checkpoint.pt')
        
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'loss': loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean up old step checkpoints (keep only the latest max_checkpoints)
        if not is_epoch_end:
            step_checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith('step_') and f.endswith('.pt')],
                key=lambda x: int(x.split('_')[1]),
                reverse=True
            )
            
            # Delete older step checkpoints beyond max_checkpoints
            for old_checkpoint in step_checkpoints[max_checkpoints:]:
                old_path = os.path.join(checkpoint_dir, old_checkpoint)
                try:
                    os.remove(old_path)
                    print(f"  Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    print(f"  Warning: Could not remove {old_checkpoint}: {e}")
    
    def compute_caption_metrics(self, preds, refs):
        """Compute captioning metrics"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            
            bleu_scores = []
            rouge_scores = []
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            smoothie = SmoothingFunction().method1
            
            for pred, ref in zip(preds, refs):
                # BLEU
                try:
                    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                    bleu_scores.append(bleu)
                except:
                    bleu_scores.append(0)
                
                # ROUGE
                try:
                    rouge = scorer.score(pred, ref)
                    rouge_scores.append(rouge['rougeL'].fmeasure)
                except:
                    rouge_scores.append(0)
            
            if bleu_scores:
                avg_bleu = np.mean(bleu_scores)
            else:
                avg_bleu = 0
            
            if rouge_scores:
                avg_rouge = np.mean(rouge_scores)
            else:
                avg_rouge = 0
            
            metrics = {
                "bleu": avg_bleu,
                "rougeL": avg_rouge,
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {"bleu": 0, "rougeL": 0}
    
    def save_model(self, save_path=None):
        """Save the trained model"""
        if save_path is None:
            save_path = self.output_dir
        
        print(f"Saving model to {save_path}...")
        try:
            # Save the model
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'msrvtt_root': self.msrvtt_root,
                'output_dir': self.output_dir,
                'device': str(self.device)
            }
            
            with open(os.path.join(save_path, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Model saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def generate_caption(self, image_path):
        """Generate caption for an image"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
                size={"height": 224, "width": 224}
            ).pixel_values.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=128,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True
                )
            
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Error generating caption"

def main():
    """Main training pipeline for MSR VTT"""
    
    print("="*80)
    print("MSR VTT VIDEO CAPTIONING FINE-TUNING")
    print("="*80)
    
    # Get MSR VTT path
    import sys
    if len(sys.argv) > 1:
        msrvtt_path = sys.argv[1]
    else:
        msrvtt_path = input("\nEnter path to MSR VTT dataset folder: ").strip().strip('"')
    
    if not os.path.exists(msrvtt_path):
        print(f"\nError: Path '{msrvtt_path}' does not exist!")
        return None
    
    # Configuration
    config = {
        'msrvtt_root': msrvtt_path,
        'model_name': 'nlpconnect/vit-gpt2-image-captioning',
        'output_dir': './msrvtt_fine_tuned_captioning',
        'extract_frames': False,  # Set to False since frames already extracted
        'frames_per_video': 1,
        'num_epochs': 3,
        'batch_size': 2,
        'learning_rate': 3e-5,
        'warmup_steps': 10,
        'val_split': 0.1,
        'use_wandb': False,
        'fp16': False,
        'use_augmentation': False,
        'use_custom_training': True  # Use custom training loop to avoid memory issues
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Process MSR VTT dataset
    print("\n" + "="*80)
    print("STEP 1: Processing MSR VTT dataset...")
    print("="*80)
    
    msrvtt_root = config['msrvtt_root']
    
    # Check folder structure
    print(f"\nChecking MSR VTT folder structure at: {msrvtt_root}")
    for item in os.listdir(msrvtt_root):
        item_path = os.path.join(msrvtt_root, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    # Check if frames already exist
    frames_dir = os.path.join(msrvtt_root, 'frames')
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') and not f.startswith('._')]
        print(f"\nFound {len(frame_files)} frames in frames directory")
    
    # Load annotations
    print("\nLoading annotations...")
    
    # First check for structured annotations
    structured_file = os.path.join(msrvtt_root, 'structured_annotations.json')
    if os.path.exists(structured_file):
        print(f"Loading existing structured annotations from {structured_file}")
        with open(structured_file, 'r', encoding='utf-8') as f:
            structured_ann = json.load(f)
        print(f"Loaded {len(structured_ann)} structured annotations")
    else:
        # Try to load from MSR_VTT.json directly
        ann_file = os.path.join(msrvtt_root, 'annotation', 'MSR_VTT.json')
        if os.path.exists(ann_file):
            print(f"Loading from {ann_file}")
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                structured_ann = []
                
                # Try different possible structures
                if isinstance(data, dict):
                    # Check for train/val/test splits
                    for split_name in ['train', 'validate', 'test', 'val']:
                        if split_name in data and isinstance(data[split_name], list):
                            print(f"Found '{split_name}' split with {len(data[split_name])} items")
                            for item in data[split_name]:
                                if isinstance(item, dict) and 'video_id' in item and 'caption' in item:
                                    structured_ann.append({
                                        'video_id': str(item['video_id']),
                                        'caption': item['caption'],
                                        'split': split_name
                                    })
                
                print(f"Parsed {len(structured_ann)} annotations from MSR_VTT.json")
                
                # Save structured annotations
                with open(structured_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_ann, f, indent=2)
                print(f"Saved structured annotations to {structured_file}")
                
            except Exception as e:
                print(f"Error loading annotations: {e}")
                structured_ann = []
        else:
            print("No annotation file found!")
            structured_ann = []
    
    # If still no annotations, create basic ones from frames
    if not structured_ann:
        print("\nCreating basic annotations from available frames...")
        frames_dir = os.path.join(msrvtt_root, 'frames')
        if os.path.exists(frames_dir):
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') and not f.startswith('._')]
            
            # Limit to 100 for testing
            if len(frame_files) > 100:
                print(f"Limiting to 100 frames (out of {len(frame_files)}) for testing")
                frame_files = frame_files[:100]
            
            structured_ann = []
            for frame_file in frame_files:
                # Extract video_id from filename
                if '_frame_' in frame_file:
                    video_id = frame_file.split('_frame_')[0]
                else:
                    video_id = frame_file.replace('.jpg', '')
                
                structured_ann.append({
                    'video_id': video_id,
                    'caption': f'A video frame showing {video_id}',
                    'split': 'train',
                    'has_frames': True,
                    'frame_files': [frame_file]
                })
            
            print(f"Created {len(structured_ann)} basic annotations from frames")
            
            # Save them
            with open(structured_file, 'w', encoding='utf-8') as f:
                json.dump(structured_ann, f, indent=2)
    
    if not structured_ann:
        print("\nERROR: No annotations available for training!")
        return None
    
    print(f"\nTotal annotations available: {len(structured_ann)}")
    
    # Step 2: Initialize finetuner
    print("\n" + "="*80)
    print("STEP 2: Initializing finetuner...")
    print("="*80)
    
    finetuner = MSRVTTFineTuner(
        msrvtt_root=msrvtt_root,
        model_name=config['model_name'],
        output_dir=config['output_dir']
    )
    
    # Step 3: Prepare datasets
    print("\n" + "="*80)
    print("STEP 3: Preparing datasets...")
    print("="*80)
    
    # Simple split - use first 80% for train, next 10% for val, last 10% for test
    if len(structured_ann) < 10:
        # Very small dataset, use all for training
        train_ann = structured_ann
        val_ann = []
        test_ann = []
    else:
        train_size = int(0.8 * len(structured_ann))
        val_size = int(0.1 * len(structured_ann))
        
        train_ann = structured_ann[:train_size]
        val_ann = structured_ann[train_size:train_size + val_size]
        test_ann = structured_ann[train_size + val_size:]
    
    print(f"Train samples: {len(train_ann)}")
    print(f"Validation samples: {len(val_ann)}")
    print(f"Test samples: {len(test_ann)}")
    
    # Create datasets
    train_dataset = finetuner.prepare_dataset(
        train_ann, 
        split='train',
        frames_per_video=1,
        use_augmentation=config['use_augmentation']
    )
    
    val_dataset = finetuner.prepare_dataset(
        val_ann, 
        split='val',
        frames_per_video=1,
        use_augmentation=False
    ) if val_ann else None
    
    test_dataset = finetuner.prepare_dataset(
        test_ann, 
        split='test',
        frames_per_video=1,
        use_augmentation=False
    ) if test_ann else None
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset) if val_dataset else 0} samples")
    print(f"  Test: {len(test_dataset) if test_dataset else 0} samples")
    
    if len(train_dataset) == 0:
        print("ERROR: No training data available!")
        return None
    
    # Step 4: Train
    print("\n" + "="*80)
    print("STEP 4: Training...")
    print("="*80)
    
    if len(train_dataset) < 5:
        print(f"Very small dataset ({len(train_dataset)} samples). Saving base model instead of training.")
        finetuner.save_model()
    else:
        # Use custom training loop to avoid memory issues
        finetuner.train_with_custom_loop(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            use_wandb=config['use_wandb'],
            fp16=config['fp16']
        )
    
    # Step 5: Test the model
    print("\n" + "="*80)
    print("STEP 5: Testing the model...")
    print("="*80)
    
    # Test with a few frames
    frames_dir = os.path.join(msrvtt_root, 'frames')
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') and not f.startswith('._')][:3]
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            try:
                caption = finetuner.generate_caption(frame_path)
                print(f"\nFrame: {frame_file}")
                print(f"Generated Caption: {caption}")
            except Exception as e:
                print(f"Error processing {frame_file}: {e}")
    
    print("\n" + "="*80)
    print("PROCESS COMPLETE!")
    print("="*80)
    print(f"Model saved to: {config['output_dir']}")
    
    # Create usage instructions
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS:")
    print("="*80)
    
    usage_code = '''
# To use the fine-tuned model in your ContentFocusedVideoAnalyzer:

# Method 1: Update the class directly
class ContentFocusedVideoAnalyzer:
    def __init__(self, use_custom_captioning=True, captioning_model_path='./msrvtt_fine_tuned_captioning'):
        if use_custom_captioning and os.path.exists(captioning_model_path):
            from transformers import pipeline
            self.image_caption_model = pipeline(
                "image-to-text", 
                model=captioning_model_path,
                use_fast=True
            )
            print("✓ Fine-tuned model loaded")
        else:
            # Use original model
            self.image_caption_model = pipeline(
                "image-to-text", 
                model="nlpconnect/vit-gpt2-image-captioning",
                use_fast=True
            )

# Method 2: Use directly
from transformers import pipeline
caption_pipeline = pipeline("image-to-text", model="./msrvtt_fine_tuned_captioning")

# Load an image and generate caption
from PIL import Image
image = Image.open("your_image.jpg")
result = caption_pipeline(image)
print(f"Caption: {result[0]['generated_text']}")
    '''
    
    print(usage_code)
    
    # Create a simple test script
    test_script = '''
import os
from transformers import pipeline
from PIL import Image

def test_finetuned_model(model_path='./msrvtt_fine_tuned_captioning'):
    """Test the fine-tuned model"""
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load the model
    print("Loading fine-tuned model...")
    caption_pipeline = pipeline("image-to-text", model=model_path)
    
    # Test with a sample image if available
    sample_images = [
        f for f in os.listdir('.') 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if sample_images:
        image = Image.open(sample_images[0])
        result = caption_pipeline(image)
        print(f"\\nImage: {sample_images[0]}")
        print(f"Generated caption: {result[0]['generated_text']}")
    else:
        print("No sample images found in current directory")
        print("Please provide an image to test the model")

if __name__ == "__main__":
    test_finetuned_model()
    '''
    
    with open('test_finetuned_model.py', 'w') as f:
        f.write(test_script)
    
    print(f"\nCreated test script: test_finetuned_model.py")
    print("Run: python test_finetuned_model.py")
    
    return finetuner

if __name__ == "__main__":
    print("="*80)
    print("MSR VTT FINE-TUNING SYSTEM")
    print("="*80)
    
    print("\nStarting fine-tuning process...")
    
    try:
        finetuner = main()
        if finetuner:
            print("\n✅ Fine-tuning completed successfully!")
        else:
            print("\n❌ Fine-tuning failed. Please check the error messages above.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()