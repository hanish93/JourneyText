# utils.py
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

def extract_key_frames(video_path, num_frames=16):
    """Extract equally spaced key frames from video"""
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

def detect_navigation_events(frames):
    """Detect turns and stops using VideoMAE model"""
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    
    # Preprocess frames
    inputs = processor(list(frames), return_tensors="pt")
    
    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def generate_vlm_description(frames, prompt):
    """Generate text description using BLIP-2 model"""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    inputs = processor(images=frames, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    
    generated_ids = model.generate(**inputs, max_new_tokens=400)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text

def summarize_journey(frames, navigation_event):
    """Generate journey summary with navigation context"""
    prompts = {
        "turn": "Describe this driving scene focusing on navigation landmarks and turning instructions. Include specific landmarks and their positions relative to the vehicle.",
        "stop": "Describe this driving scene focusing on reasons for stopping and landmarks near the stopping point. Include traffic elements and road features.",
        "straight": "Describe this driving scene focusing on the path ahead and notable landmarks along the road. Include distance estimates and upcoming features.",
        "default": "Describe this driving scene for navigation purposes, highlighting important landmarks, road features, and navigation instructions."
    }
    
    prompt = prompts.get(navigation_event.split()[0].lower(), prompts["default"])
    return generate_vlm_description(frames, prompt)

def process_video(video_path):
    """Main processing pipeline"""
    # Step 1: Extract key frames
    frames = extract_key_frames(video_path)
    
    # Step 2: Detect navigation events
    event = detect_navigation_events(frames)
    
    # Step 3: Generate journey summary
    summary = summarize_journey(frames, event)
    
    return {
        "event": event,
        "summary": summary,
        "frames": frames
    }