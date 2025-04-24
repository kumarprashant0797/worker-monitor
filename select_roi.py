# python select_roi.py --cam <cam_url> --num <num> 
# --cam: camera url or video file
# --num: number of points in roi. [optional, default: 4]
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--cam', help='Camera index or video file path')
ap.add_argument('--num', type=int, default=4, help='No. of points')
ap.add_argument('--config', default='config.json', help='Path to config file')
args = ap.parse_args()

cam_url = args.cam
if cam_url in ['0', '1']:
    cam_url = int(cam_url)

# Open the video source
cap = cv2.VideoCapture(cam_url)
ret, frame = cap.read()
if not ret:
    print(f"Error: Could not read from {cam_url}")
    exit(1)
cap.release()

# Convert to RGB for matplotlib
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the image and let user select points
plt.figure(figsize=(12, 8))
plt.imshow(frame)
plt.title("Select ROI points (click to select points)")
pts = plt.ginput(args.num, timeout=-1)
plt.close()  # Automatically close the window after n points are selected

# Convert points to integer coordinates
pts = np.array(pts, dtype=int)

# Calculate bounding box of the selected points
x_min = min(pts[:, 0])
y_min = min(pts[:, 1])
x_max = max(pts[:, 0])
y_max = max(pts[:, 1])

# Calculate width and height
width = x_max - x_min
height = y_max - y_min

# Format for config.json
roi_config = {
    "roi_x": int(x_min),
    "roi_y": int(y_min),
    "roi_width": int(width),
    "roi_height": int(height)
}

print("\nSelected ROI points:", pts.tolist())
print("\nConfig values for config.json:")
print(f"  \"roi_x\": {roi_config['roi_x']},")
print(f"  \"roi_y\": {roi_config['roi_y']},")
print(f"  \"roi_width\": {roi_config['roi_width']},")
print(f"  \"roi_height\": {roi_config['roi_height']},")

# Ask if user wants to update config.json
update_config = input("\nDo you want to update config.json with these values? (y/n): ")
if update_config.lower() == 'y':
    try:
        # Load existing config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update ROI values
        config['roi_x'] = roi_config['roi_x']
        config['roi_y'] = roi_config['roi_y']
        config['roi_width'] = roi_config['roi_width']
        config['roi_height'] = roi_config['roi_height']
        
        # Save updated config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Successfully updated {args.config} with new ROI values.")
    except Exception as e:
        print(f"Error updating config file: {e}")

# Only show the final visualization if user wants to see it
show_viz = input("\nDo you want to see the selected ROI? (y/n): ")
if show_viz.lower() == 'y':
    # Visualize the selected ROI on the image
    cv2.rectangle(frame, 
                (roi_config['roi_x'], roi_config['roi_y']),
                (roi_config['roi_x'] + roi_config['roi_width'], roi_config['roi_y'] + roi_config['roi_height']),
                (255, 0, 0), 2)

    # Display image with the ROI rectangle
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.title("Selected ROI")
    plt.show()
    print("Returned to terminal after showing visualization.")
