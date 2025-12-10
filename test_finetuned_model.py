
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
        print(f"\nImage: {sample_images[0]}")
        print(f"Generated caption: {result[0]['generated_text']}")
    else:
        print("No sample images found in current directory")
        print("Please provide an image to test the model")

if __name__ == "__main__":
    test_finetuned_model()
    