# utils.py (CPU optimized)
import cv2
import decord
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
import warnings

# Suppress some verbose warnings
warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size` from.*")

def extract_key_frames(video_path, num_frames=8):  # Reduced from 16 to 8 for CPU
    """Extract equally spaced key frames from video"""
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError("Video has no frames")
            
        indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

def detect_navigation_events(frames):
    """Detect turns and stops using lightweight VideoMAE model"""
    if not frames:
        return "no frames available"
    
    try:
        # Use smaller model for CPU
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")
        
        # Select fewer frames for CPU processing
        selected_frames = frames[:4]  # Process only first 4 frames
        
        # Preprocess frames
        inputs = processor(selected_frames, return_tensors="pt")
        
        # Run model inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        return model.config.id2label[predicted_class_idx]
    except Exception as e:
        print(f"Error detecting events: {e}")
        return "unknown"

def generate_vlm_description(frames, prompt):
    """Generate text description using smaller BLIP-2 model for CPU"""
    if not frames:
        return "No frames available for description"
    
    try:
        # Use smaller model for CPU
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float32  # Use float32 instead of float16 for CPU
        )
        
        # Process only the first frame for CPU efficiency
        frame = frames[0]
        
        inputs = processor(images=frame, text=prompt, return_tensors="pt")
        
        generated_ids = model.generate(**inputs, max_new_tokens=200)  # Reduced from 200
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return generated_text
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Could not generate description"

def summarize_journey(frames, navigation_event):
    """Generate journey summary with navigation context (CPU optimized)"""
    prompts = {
        "turn": "Describe landmarks for navigation. Focus on turns.",
        "stop": "Describe why stopping. Mention nearby objects.",
        "straight": "Describe the road ahead briefly.",
        "default": "Describe the scene for navigation purposes."
    }
    
    prompt = prompts.get(navigation_event.split()[0].lower(), prompts["default"])
    return generate_vlm_description(frames, prompt)

def process_video(video_path):
    """Main processing pipeline (CPU optimized)"""
    try:
        # Step 1: Extract key frames (fewer frames for CPU)
        frames = extract_key_frames(video_path, num_frames=8)
        
        if not frames:
            return {
                "event": "video processing error",
                "summary": "Could not extract frames from video",
                "frames": []
            }
        
        # Step 2: Detect navigation events
        event = detect_navigation_events(frames)
        
        # Step 3: Generate journey summary
        summary = summarize_journey(frames, event)
        
        return {
            "event": event,
            "summary": summary,
            "frames": frames[:1]  # Return only first frame for display
        }
    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            "event": "processing error",
            "summary": f"Error processing video: {str(e)}",
            "frames": []
        }