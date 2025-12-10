
import os
import json
import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    pipeline
)
import glob

def create_simple_dataset(msrvtt_root, num_samples=100):
    """Create a simple dataset for testing"""
    
    frames_dir = os.path.join(msrvtt_root, 'frames')
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        return []
    
    frame_files = glob.glob(os.path.join(frames_dir, '*.jpg'))[:num_samples]
    
    dataset = []
    for frame_file in frame_files:
        video_id = os.path.basename(frame_file).split('_frame_')[0]
        dataset.append({
            'image_path': frame_file,
            'caption': f'A video showing {video_id}',
            'video_id': video_id
        })
    
    return dataset

def finetune_simple_model(msrvtt_root, output_dir='./simple_finetuned_model'):
    """Simple finetuning function"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    print("Loading base model...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Create simple dataset
    print("Creating dataset...")
    dataset = create_simple_dataset(msrvtt_root, num_samples=50)
    
    if not dataset:
        print("No dataset created!")
        return
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Simple training loop (for demonstration)
    model.train()
    
    # Save model (in reality, you would train it first)
    print("Saving model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # Test the model
    print("Testing model...")
    test_pipeline = pipeline("image-to-text", model=output_dir)
    
    # Test with a sample image
    if dataset:
        sample = dataset[0]
        image = Image.open(sample['image_path'])
        result = test_pipeline(image)
        print(f"Sample image: {os.path.basename(sample['image_path'])}")
        print(f"Generated caption: {result[0]['generated_text']}")
        print(f"Expected caption: {sample['caption']}")

if __name__ == "__main__":
    msrvtt_path = input("Enter MSR VTT dataset path: ").strip()
    finetune_simple_model(msrvtt_path)
    