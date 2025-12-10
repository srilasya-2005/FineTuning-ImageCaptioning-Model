"""
Resume training from a checkpoint.
Loads the model state from a checkpoint and continues training.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)
from PIL import Image
import glob
import random

# Import the dataset class from finetune.py
import sys
sys.path.insert(0, '.')
from finetune import MSRVTTCaptioningDataset

def resume_training(
    checkpoint_path='msrvtt_fine_tuned_captioning/checkpoints/step_1350_checkpoint.pt',
    msrvtt_root='dataset/MSRVTT',
    output_dir='./msrvtt_fine_tuned_captioning',
    num_epochs=3,
    batch_size=2,
    learning_rate=3e-5,
    save_steps=50,
    device='cuda'
):
    """Resume training from a checkpoint"""
    
    print("=" * 60)
    print("RESUMING MSR-VTT FINE-TUNING")
    print("=" * 60)
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    start_epoch = checkpoint.get('epoch', 0)
    start_step = checkpoint.get('step', 0)
    last_loss = checkpoint.get('loss', 0)
    
    print(f"  Resuming from epoch {start_epoch}, step {start_step}")
    print(f"  Last loss was: {last_loss:.4f}")
    
    # Load model
    print("\nLoading base model...")
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens (same as training)
    special_tokens_dict = {
        'additional_special_tokens': [
            '<video>', '</video>',
            '<scene>', '</scene>',
            '<action>', '</action>',
            '<person>', '</person>',
            '<object>', '</object>'
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"  Added {num_added_toks} special tokens")
    
    # Resize embeddings
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    # Load checkpoint weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("  Model loaded successfully!")
    
    # Load annotations
    ann_path = os.path.join(msrvtt_root, 'structured_annotations.json')
    print(f"\nLoading annotations from: {ann_path}")
    with open(ann_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Filter for training data
    train_annotations = [a for a in annotations if a.get('split') == 'train']
    print(f"  Train annotations: {len(train_annotations)}")
    
    # Create dataset
    print("\nPreparing dataset...")
    dataset = MSRVTTCaptioningDataset(
        msrvtt_root=msrvtt_root,
        annotations=train_annotations,
        processor=processor,
        tokenizer=tokenizer,
        max_length=128,
        image_size=224,
        frames_per_video=1,
        use_augmentation=False
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: {
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    remaining_steps = total_steps - start_step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Starting from step: {start_step}")
    print(f"Total steps per epoch: {len(train_loader)}")
    print(f"Epochs: {num_epochs}")
    
    global_step = start_step
    model.train()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        # Skip already processed batches in the first epoch
        skip_batches = 0
        if epoch == start_epoch:
            skip_batches = start_step % len(train_loader)
            if skip_batches > 0:
                print(f"  Skipping first {skip_batches} batches (already processed)")
        
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in train_progress:
            # Skip already processed batches
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            
            try:
                # Move batch to device
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
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
                avg_loss = epoch_loss / (batch_idx - skip_batches + 1)
                
                train_progress.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Save checkpoint
                global_step += 1
                if global_step % save_steps == 0:
                    checkpoint_save_path = os.path.join(checkpoint_dir, f'step_{global_step}_checkpoint.pt')
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_save_path)
                    print(f"\n  Checkpoint saved: {checkpoint_save_path}")
                    
                    # Clean up old step checkpoints (keep only latest 2)
                    step_checkpoints = sorted(
                        [f for f in os.listdir(checkpoint_dir) if f.startswith('step_') and f.endswith('.pt')],
                        key=lambda x: int(x.split('_')[1]),
                        reverse=True
                    )
                    for old_ckpt in step_checkpoints[2:]:
                        try:
                            os.remove(os.path.join(checkpoint_dir, old_ckpt))
                            print(f"  Removed old checkpoint: {old_ckpt}")
                        except:
                            pass
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        # End of epoch checkpoint
        avg_epoch_loss = epoch_loss / (len(train_loader) - skip_batches)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Epoch checkpoint saved: {checkpoint_path}")
    
    # Save final model
    print(f"\n{'='*60}")
    print("SAVING FINAL MODEL")
    print(f"{'='*60}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    print("\nTraining complete!")
    
    return model

if __name__ == "__main__":
    # Resume from step 1350
    resume_training(
        checkpoint_path='msrvtt_fine_tuned_captioning/checkpoints/step_1350_checkpoint.pt',
        msrvtt_root='dataset/MSRVTT',
        num_epochs=3,
        batch_size=2,
        learning_rate=3e-5,
        save_steps=50
    )
