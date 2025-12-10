"""
Evaluate the fine-tuned MSR-VTT captioning model.
Loads a checkpoint and evaluates on test set using BLEU and ROUGE-L metrics.
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)
from tqdm import tqdm
import random

# Metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "nltk", "rouge-score"])
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

def load_annotations(ann_path, split='test'):
    """Load annotations for a specific split"""
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered = [d for d in data if d.get('split') == split]
    print(f"Loaded {len(filtered)} {split} annotations")
    return filtered

def load_checkpoint(checkpoint_path, model_name="nlpconnect/vit-gpt2-image-captioning", device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load base model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add the same special tokens used during training
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
    print(f"  Added {num_added_toks} special tokens to match training")
    
    # Resize model embeddings to match checkpoint
    model.decoder.resize_token_embeddings(len(tokenizer))
    print(f"  Tokenizer size: {len(tokenizer)}")
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('step', 'unknown')}")
        print(f"  Training loss was: {checkpoint.get('loss', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, processor, tokenizer

def generate_caption(model, processor, tokenizer, image_path, device='cuda', max_length=50):
    """Generate a caption for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=3,
                early_stopping=True
            )
        
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return ""

def compute_metrics(predictions, references):
    """Compute BLEU and ROUGE-L scores"""
    bleu_scores = []
    rouge_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        # BLEU
        try:
            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0)
        
        # ROUGE-L
        try:
            rouge = scorer.score(ref, pred)
            rouge_scores.append(rouge['rougeL'].fmeasure)
        except:
            rouge_scores.append(0)
    
    return {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0,
        'rouge_l': np.mean(rouge_scores) if rouge_scores else 0,
        'num_samples': len(predictions)
    }

def evaluate_model(
    checkpoint_path,
    annotations_path,
    frames_dir,
    split='test',
    max_samples=100,
    device='cuda'
):
    """Main evaluation function"""
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("MSR-VTT Model Evaluation")
    print("=" * 60)
    
    # Load annotations
    annotations = load_annotations(annotations_path, split=split)
    
    if max_samples and len(annotations) > max_samples:
        print(f"Sampling {max_samples} videos for evaluation...")
        annotations = random.sample(annotations, max_samples)
    
    # Load model
    model, processor, tokenizer = load_checkpoint(checkpoint_path, device=device)
    
    # Generate captions
    print(f"\nGenerating captions for {len(annotations)} videos...")
    predictions = []
    references = []
    results_detail = []
    
    for ann in tqdm(annotations):
        video_id = ann['video_id']
        ref_caption = ann['caption']
        frame_files = ann.get('frame_files', [])
        
        if not frame_files:
            continue
        
        # Use first frame for captioning
        frame_path = os.path.join(frames_dir, frame_files[0])
        
        if not os.path.exists(frame_path):
            continue
        
        pred_caption = generate_caption(model, processor, tokenizer, frame_path, device=device)
        
        predictions.append(pred_caption)
        references.append(ref_caption)
        results_detail.append({
            'video_id': video_id,
            'reference': ref_caption,
            'prediction': pred_caption
        })
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, references)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"BLEU Score:        {metrics['bleu']:.4f}")
    print(f"ROUGE-L Score:     {metrics['rouge_l']:.4f}")
    print("=" * 60)
    
    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 60)
    for i, result in enumerate(results_detail[:5]):
        print(f"\n[{result['video_id']}]")
        print(f"  Reference:  {result['reference'][:80]}...")
        print(f"  Prediction: {result['prediction'][:80]}...")
    
    # Save detailed results
    results_file = 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'details': results_detail
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return metrics, results_detail

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MSR-VTT captioning model')
    parser.add_argument('--checkpoint', type=str, 
                        default='msrvtt_fine_tuned_captioning/checkpoints/step_4550_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--annotations', type=str,
                        default='dataset/MSRVTT/structured_annotations.json',
                        help='Path to annotations file')
    parser.add_argument('--frames_dir', type=str,
                        default='dataset/MSRVTT/frames',
                        help='Path to frames directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to evaluate on (train/val/test)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum samples to evaluate (0 for all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        annotations_path=args.annotations,
        frames_dir=args.frames_dir,
        split=args.split,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        device=args.device
    )
