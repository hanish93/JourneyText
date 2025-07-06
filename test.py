# test.py
import os
import cv2
import numpy as np
import time
from utils import extract_key_frames, detect_navigation_events, generate_vlm_description

def create_test_video(output_path, duration=2, fps=10, width=640, height=480):
    """Create a synthetic test video with simple patterns"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create frames with moving patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i % width, 0), ((i+100) % width, height//2), (0, 255, 0), -1)
        cv2.circle(frame, (width//2, height//2), 50 + i % 100, (255, 0, 0), 2)
        out.write(frame)
    
    out.release()
    return output_path

def run_tests():
    print("🚀 Starting comprehensive tests...")
    test_results = {"passed": 0, "failed": 0}
    test_video = "test_video.mp4"
    
    try:
        # Test 1: Video creation
        print("\n🧪 Test 1: Creating test video...")
        create_test_video(test_video)
        if os.path.exists(test_video):
            print("✅ Test 1 passed: Video created successfully")
            test_results["passed"] += 1
        else:
            print("❌ Test 1 failed: Video not created")
            test_results["failed"] += 1
            return test_results
        
        # Test 2: Frame extraction
        print("\n🧪 Test 2: Frame extraction...")
        frames = extract_key_frames(test_video, num_frames=8)
        if frames and len(frames) > 0:
            print(f"✅ Test 2 passed: Extracted {len(frames)} frames")
            test_results["passed"] += 1
        else:
            print("❌ Test 2 failed: No frames extracted")
            test_results["failed"] += 1
            return test_results
        
        # Test 3: Event detection
        print("\n🧪 Test 3: Navigation event detection...")
        event = detect_navigation_events(frames)
        if event != "error" and event != "no-frames":
            print(f"✅ Test 3 passed: Detected event '{event}'")
            test_results["passed"] += 1
        else:
            print(f"❌ Test 3 failed: Event detection error")
            test_results["failed"] += 1
        
        # Test 4: Description generation
        print("\n🧪 Test 4: Description generation...")
        description = generate_vlm_description(frames, "Describe this scene")
        if description and "error" not in description.lower():
            print(f"✅ Test 4 passed: Generated description")
            print(f"   Sample: {description[:80]}...")
            test_results["passed"] += 1
        else:
            print(f"❌ Test 4 failed: Description generation error")
            test_results["failed"] += 1
        
        # Test 5: Full pipeline
        print("\n🧪 Test 5: Full processing pipeline...")
        from utils import process_video
        start_time = time.time()
        result = process_video(test_video)
        elapsed = time.time() - start_time
        
        if "error" not in result:
            print(f"✅ Test 5 passed: Processed in {elapsed:.2f}s")
            print(f"   Event: {result['event']}")
            print(f"   Summary: {result['summary'][:80]}...")
            test_results["passed"] += 1
        else:
            print(f"❌ Test 5 failed: {result.get('error', 'Unknown error')}")
            test_results["failed"] += 1
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        test_results["failed"] += 1
    finally:
        if os.path.exists(test_video):
            os.remove(test_video)
    
    return test_results

if __name__ == "__main__":
    results = run_tests()
    print("\n📊 Test Results:")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"🚀 Success Rate: {results['passed']/(results['passed']+results['failed'])*100:.0f}%")
    
    if results["failed"] == 0:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check logs for details.")
        sys.exit(1)