# app.py
import streamlit as st
import os
import time
import torch
from utils import process_video

# Configure Streamlit
st.set_page_config(
    page_title="Journey Summarizer",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stVideo { border-radius: 10px; }
    .summary-box { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: #f0f2f6; 
        margin-top: 20px;
    }
    .highlight { background-color: #fffacd; padding: 2px 5px; border-radius: 3px; }
    .error-box { background-color: #ffebee; color: #b71c1c; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

def display_results(result):
    """Display processing results"""
    if "error" in result:
        st.error("Processing Error")
        st.markdown(f'<div class="error-box">{result["error"]}</div>', unsafe_allow_html=True)
        return
        
    st.subheader("Navigation Analysis")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Detected Event", result["event"])
        if result.get("frames") and len(result["frames"]) > 0:
            st.image(result["frames"][0], caption="Key Frame", use_column_width=True)
        else:
            st.warning("No frames available for display")
    
    with col2:
        st.subheader("Journey Summary")
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
        
        if "error" not in result["summary"].lower():
            st.subheader("Navigation Instructions")
            instructions = [s.strip() for s in result["summary"].split(". ") if s.strip()]
            if instructions:
                for i, instruction in enumerate(instructions):
                    st.markdown(f"{i+1}. <span class='highlight'>{instruction}</span>", 
                               unsafe_allow_html=True)
            else:
                st.info("No instructions generated")

# App header
st.title("🚗 Journey Summarization Using Visual-Language Models")
st.markdown("Upload driving videos to generate AI-powered navigation summaries")
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Upload driving video (MP4, AVI, MOV)", 
    type=["mp4", "avi", "mov"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        # Save uploaded file
        video_path = f"temp_{int(time.time())}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video
        st.video(video_path)
        
        # Process video
        with st.spinner("Analyzing video content..."):
            result = process_video(video_path)
            display_results(result)
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# CLI instructions
st.sidebar.header("CLI Processing")
st.sidebar.markdown("""
**Process videos via CLI while Docker is running:**
```bash
# Process a video file
docker exec -it journey-container \\
  python cli.py /data/videos/input.mp4 \\
  --output /data/results/output.json

# Mount custom directories
docker run -d ... \\
  -v /host/videos:/data/videos \\
  -v /host/results:/data/results""")