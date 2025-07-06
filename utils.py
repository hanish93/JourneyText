# utils.py
import os
import cv2
import torch
import numpy as np
import logging
from decord import VideoReader, gpu, cpu
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    Blip2Processor,
    Blip2ForConditionalGeneration
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FRAME_SIZE = (224, 224)
BLIP2_FRAME_SIZE = (384, 384)
MODEL_CACHE_DIR = "/app/model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Preload models and processors
try:
    logger.info("Loading VideoMAE processor and model...")
    video_processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        cache_dir=MODEL_CACHE_DIR,
        use_fast=True
    )
    video_model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        cache_dir=MODEL_CACHE_DIR
    ).to(device)
    video_model.eval()
    
    logger.info("Loading BLIP-2 processor and model...")
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir=MODEL_CACHE_DIR,
        use_fast=True
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)
    blip_model.eval()
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise RuntimeError("Model loading failed") from e

def extract_key_frames(video_path, num_frames=16):
    """Robust frame extraction with GPU acceleration"""
    try:
        # Use GPU decoding if available
        ctx = gpu(0) if device.type == 'cuda' else cpu(0)
        vr = VideoReader(video_path, ctx=ctx)
        
        total_frames = len(vr)
        if total_frames == 0:
            logger.error("Video has 0 frames")
            return []
            
        # Calculate frame indices
        num_frames = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
        
        # Extract and resize frames
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, DEFAULT_FRAME_SIZE)
            frames.append(frame)
            
        return frames
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return []

def detect_navigation_events(frames):
    """Event detection with automatic frame padding"""
    try:
        # Validate input
        if len(frames) == 0:
            return "no-frames"
            
        # Pad or truncate to 16 frames
        if len(frames) < 16:
            frames = frames + [frames[-1]] * (16 - len(frames))
        elif len(frames) > 16:
            frames = frames[:16]
        
        # Prepare inputs
        inputs = video_processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = video_model(**inputs)
            else:
                outputs = video_model(**inputs)
                
        return video_model.config.id2label[outputs.logits.argmax().item()]
    except Exception as e:
        logger.error(f"Navigation detection failed: {str(e)}")
        return "error"

def generate_vlm_description(frames, prompt):
    """Generate description with first frame only"""
    try:
        if len(frames) == 0:
            return "no-frames"
            
        # Process only the first frame
        frame = cv2.resize(frames[0], BLIP2_FRAME_SIZE)
        
        # Prepare inputs
        inputs = blip_processor(
            images=frame,
            text=prompt,
            return_tensors="pt"
        ).to(device, torch.float16 if device.type == 'cuda' else torch.float32)
        
        # Generate text
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    generated_ids = blip_model.generate(**inputs, max_new_tokens=200)
            else:
                generated_ids = blip_model.generate(**inputs, max_new_tokens=200)
                
        return blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Description generation failed: {str(e)}")
        return "error"

def summarize_journey(frames, navigation_event):
    """Generate context-aware summary"""
    prompts = {
        "turn": "Describe landmarks and turning instructions for navigation.",
        "stop": "Describe reasons for stopping and nearby landmarks.",
        "straight": "Describe the road ahead and upcoming landmarks.",
        "default": "Describe this driving scene for navigation purposes."
    }
    
    event_type = navigation_event.split()[0].lower() if navigation_event else "default"
    prompt = prompts.get(event_type, prompts["default"])
    return generate_vlm_description(frames, prompt)

def process_video(video_path):
    """Full processing pipeline with error handling"""
    try:
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Extract frames
        frames = extract_key_frames(video_path)
        if len(frames) == 0:
            return {"error": "Frame extraction failed"}
            
        # Step 2: Detect navigation event
        event = detect_navigation_events(frames)
        
        # Step 3: Generate summary
        summary = summarize_journey(frames, event)
        
        return {
            "event": event,
            "summary": summary,
            "frames": frames
        }
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        return {"error": str(e)}