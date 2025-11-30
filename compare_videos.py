import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
BEFORE_VIDEO_PATH = "fix.mp4"  # Assuming this is the clean/before video
AFTER_VIDEO_PATH = "dent.mp4"  # Assuming this is the damaged/after video
MODEL_PATH = "yolov8n.pt"      # REPLACE with your trained model path, e.g., 'runs/detect/train/weights/best.pt'
OUTPUT_DIR = "comparison_results"
FRAME_INTERVAL = 30            # Process 1 frame every 30 frames (approx 1 sec)

def extract_frames(video_path, interval=30):
    """Extracts frames from a video at a specific interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def find_best_match(query_frame, database_frames):
    """Finds the most similar frame in database_frames using ORB feature matching."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(query_frame, None)
    
    if des1 is None:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_match_frame = None
    max_matches = -1
    
    for db_frame in database_frames:
        kp2, des2 = orb.detectAndCompute(db_frame, None)
        if des2 is None:
            continue
            
        matches = bf.match(des1, des2)
        if len(matches) > max_matches:
            max_matches = len(matches)
            best_match_frame = db_frame
            
    return best_match_frame, max_matches

def main():
    # 1. Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    # 2. Extract Frames
    print("Processing 'Before' video...")
    before_frames = extract_frames(BEFORE_VIDEO_PATH, FRAME_INTERVAL)
    
    print("Processing 'After' video...")
    # We process 'After' video frame by frame (with interval) for detection
    cap_after = cv2.VideoCapture(AFTER_VIDEO_PATH)
    
    frame_count = 0
    damage_count = 0
    
    print("Starting comparison...")
    while cap_after.isOpened():
        ret, frame_after = cap_after.read()
        if not ret:
            break
            
        if frame_count % FRAME_INTERVAL == 0:
            # 3. Detect Damages
            results = model(frame_after, verbose=False)
            
            # Check if any damage detected
            detections = results[0].boxes
            if len(detections) > 0:
                print(f"Damage detected at frame {frame_count}")
                
                # Draw bounding boxes on the 'After' frame
                annotated_frame_after = results[0].plot()
                
                # 4. Find Match in 'Before' video
                matched_frame_before, score = find_best_match(frame_after, before_frames)
                
                if matched_frame_before is not None:
                    # Resize for side-by-side if dimensions differ
                    h1, w1 = annotated_frame_after.shape[:2]
                    h2, w2 = matched_frame_before.shape[:2]
                    
                    if h1 != h2 or w1 != w2:
                        matched_frame_before = cv2.resize(matched_frame_before, (w1, h1))
                    
                    # 5. Create Comparison Image
                    # Add labels
                    cv2.putText(matched_frame_before, "BEFORE (Reference)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame_after, "AFTER (Damage Detected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Concatenate
                    comparison = np.hstack((matched_frame_before, annotated_frame_after))
                    
                    # Save
                    output_filename = os.path.join(OUTPUT_DIR, f"damage_comparison_{damage_count}.jpg")
                    cv2.imwrite(output_filename, comparison)
                    print(f"  Saved comparison to {output_filename}")
                    damage_count += 1
                else:
                    print("  Could not find a matching frame in 'Before' video.")

        frame_count += 1
        
    cap_after.release()
    print("Done.")

if __name__ == "__main__":
    main()
