# app.py (CPU optimized)
import streamlit as st
import os
import time
from utils import process_video

# Configure Streamlit
st.set_page_config(
    page_title="Journey Summarizer (CPU)",
    page_icon="🚗",
    layout="centered"  # Changed from wide to centered for better mobile view
)

# Custom CSS
st.markdown("""
    <style>
    .stVideo { border-radius: 10px; max-width: 100%; }
    .summary-box { 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #f0f2f6; 
        margin: 10px 0;
        font-size: 0.9em;
    }
    .highlight { 
        background-color: #fffacd; 
        padding: 2px 5px; 
        border-radius: 3px;
        font-size: 0.9em;
    }
    .stSpinner > div { margin: 0 auto; }
    .small-text { font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🚗 Journey Summarization (CPU Version)")
st.markdown("""
    <p class="small-text">Upload a driving video to generate navigation instructions</p>
    """, unsafe_allow_html=True)
st.divider()

# File uploader with size limit
MAX_MB = 200
uploaded_file = st.file_uploader(
    f"Upload driving video (MP4, max {MAX_MB}MB)", 
    type=["mp4"],
    accept_multiple_files=False
)

if uploaded_file:
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # in MB
    if file_size > MAX_MB:
        st.error(f"File too large. Maximum size is {MAX_MB}MB")
        st.stop()
    
    # Save uploaded file
    video_path = f"./temp_{int(time.time())}.mp4"
    try:
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video
        st.video(video_path)
        
        # Process video with progress
        with st.spinner("Analyzing video (this may take a minute)..."):
            result = process_video(video_path)
        
        # Display results
        st.subheader("Navigation Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Detected Event", result["event"])
            if result["frames"]:
                st.image(result["frames"][0], caption="Key Frame", use_container_width=True)
        
        with col2:
            st.subheader("Journey Summary")
            st.markdown(f'<div class="summary-box">{result["summary"]}</div>', 
                       unsafe_allow_html=True)
            
            # Generate navigation instructions
            st.subheader("Key Instructions")
            instructions = [s.strip() for s in result["summary"].split(". ") if s.strip()]
            for i, instruction in enumerate(instructions[:3]):  # Show max 3 instructions
                st.markdown(f"{i+1}. <span class='highlight'>{instruction}</span>", 
                           unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)

# Add sidebar information
st.sidebar.header("About (CPU Version)")
st.sidebar.markdown("""
<p class="small-text">
This CPU-optimized version uses smaller models and processes fewer frames to run without GPU.
</p>

**Limitations:**
- Processes only key frames
- Simplified descriptions
- Longer processing time

**Tips:**
- Use short videos (<30 sec)
- Landscape orientation works best
- Well-lit scenes give better results
""", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.info("Upload a short driving video to begin")