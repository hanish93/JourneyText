
# cli.py
import argparse
import json
import os
import sys
import time
from utils import process_video

def main():
    parser = argparse.ArgumentParser(description='Journey Summarization CLI')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON path')
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video_path):
        print(f"Error: File not found - {args.video_path}")
        sys.exit(1)
        
    print(f"🚗 Processing: {args.video_path}")
    start_time = time.time()
    
    try:
        result = process_video(args.video_path)
        elapsed = time.time() - start_time
        
        # Handle errors
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            sys.exit(1)
            
        # Save results
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Results saved to: {args.output}")
        print(f"⏱️ Processing time: {elapsed:.2f} seconds")
        print("\n📝 Journey Summary:")
        print(result['summary'])
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()