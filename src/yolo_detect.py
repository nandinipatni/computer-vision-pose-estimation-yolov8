from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import json
from pathlib import Path

def get_image_files(path):
    """Get image files from path (file or directory)"""
    path = Path(path)
    
    if path.is_file():
        return [path]
    elif path.is_dir():
        # Get all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
            image_files.extend(path.glob(f'*{ext.upper()}'))
        return sorted(image_files)
    else:
        raise ValueError(f"Path does not exist: {path}")

def save_keypoints_json(image_path, result, save_path):
    """Save keypoint coordinates as percentages to JSON file"""
    # Get image dimensions
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    
    # Initialize data structure
    json_data = {
        "image_name": Path(image_path).name,
        "image_width": int(img_width),
        "image_height": int(img_height),
        "objects": []
    }
    
    # Extract keypoints if available
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()  # Get x,y coordinates
        confidence = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
        
        for obj_idx, obj_keypoints in enumerate(keypoints):
            obj_data = {
                "object_id": obj_idx + 1,
                "keypoints": []
            }
            
            # Process each keypoint
            for kp_idx, (x, y) in enumerate(obj_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    # Convert to percentages and ensure Python native types
                    x_percent = float((x / img_width) * 100)
                    y_percent = float((y / img_height) * 100)
                    
                    kp_data = {
                        "keypoint_id": kp_idx + 1,
                        "x_pixel": float(x),
                        "y_pixel": float(y),
                        "x_percent": round(x_percent, 2),
                        "y_percent": round(y_percent, 2),
                        "confidence": float(confidence[obj_idx][kp_idx]) if confidence is not None else 1.0
                    }
                    obj_data["keypoints"].append(kp_data)
            
            if obj_data["keypoints"]:  # Only add objects with valid keypoints
                json_data["objects"].append(obj_data)
    
    # Add bounding boxes if available
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            
            # Find corresponding object or create new one
            if i < len(json_data["objects"]):
                json_data["objects"][i]["bounding_box"] = {
                    "x1_pixel": float(x1),
                    "y1_pixel": float(y1),
                    "x2_pixel": float(x2),
                    "y2_pixel": float(y2),
                    "x1_percent": round(float((x1 / img_width) * 100), 2),
                    "y1_percent": round(float((y1 / img_height) * 100), 2),
                    "x2_percent": round(float((x2 / img_width) * 100), 2),
                    "y2_percent": round(float((y2 / img_height) * 100), 2),
                    "confidence": float(conf)
                }
    
    # Save JSON file
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return json_data

def visualize_yolo_results(image_path, result, save_path=None, show_plot=False):
    """Visualize YOLO keypoint detection results"""
    # Load original image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Extract keypoints if available
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()  # Get x,y coordinates
        confidence = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
        
        # Plot keypoints for each detected object
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for obj_idx, obj_keypoints in enumerate(keypoints):
            color = colors[obj_idx % len(colors)]
            
            # Plot each keypoint
            for kp_idx, (x, y) in enumerate(obj_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    alpha = 1.0
                    if confidence is not None:
                        alpha = float(confidence[obj_idx][kp_idx])
                    
                    plt.scatter(x, y, c=[color], s=100, alpha=alpha, 
                              edgecolors='white', linewidth=2)
                    plt.annotate(f'P{kp_idx+1}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', color='white', 
                               fontweight='bold', fontsize=8)
    
    # Check for bounding boxes
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-10, f'Conf: {conf:.2f}', color='red', fontweight='bold')
    
    plt.title(f'YOLO Keypoint Detection - {Path(image_path).name}')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot to free memory
    
    return result
    """Visualize YOLO keypoint detection results"""
    # Load original image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Extract keypoints if available
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()  # Get x,y coordinates
        confidence = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
        
        # Plot keypoints for each detected object
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for obj_idx, obj_keypoints in enumerate(keypoints):
            color = colors[obj_idx % len(colors)]
            
            # Plot each keypoint
            for kp_idx, (x, y) in enumerate(obj_keypoints):
                if x > 0 and y > 0:  # Valid keypoint
                    alpha = 1.0
                    if confidence is not None:
                        alpha = float(confidence[obj_idx][kp_idx])
                    
                    plt.scatter(x, y, c=[color], s=100, alpha=alpha, 
                              edgecolors='white', linewidth=2)
                    plt.annotate(f'P{kp_idx+1}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', color='white', 
                               fontweight='bold', fontsize=8)
    
    # Check for bounding boxes
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-10, f'Conf: {conf:.2f}', color='red', fontweight='bold')
    
    plt.title(f'YOLO Keypoint Detection - {Path(image_path).name}')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot to free memory
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run YOLO point detection inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file (best.pt)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output images')
    parser.add_argument('--show-plots', action='store_true', help='Show matplotlib plots (default: False for batch processing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Load model
    try:
        print(f"Loading YOLO model from: {args.model}")
        model = YOLO(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get image files
    try:
        image_files = get_image_files(args.image)
        # Remove duplicates (you had duplicate entries)
        image_files = list(dict.fromkeys(image_files))
        print(f"Found {len(image_files)} unique image file(s)")
        for img_file in image_files:
            print(f"  - {img_file.name}")
    except Exception as e:
        print(f"Error finding images: {e}")
        return
    
    # Initialize summary
    total_processed = 0
    total_objects = 0
    total_keypoints = 0
    
    # Process all images
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            # Run inference
            results = model.predict(
                source=str(image_path),
                save=False,
                show=False,
                conf=0.25,  # confidence threshold
                verbose=False
            )
            result = results[0]  # Get first result
            
            # Generate output paths
            output_name = f"result_{image_path.stem}.png"
            json_name = f"keypoints_{image_path.stem}.json"
            output_path = output_dir / output_name
            json_path = output_dir / json_name
            
            # Visualize results (without showing plots for batch processing)
            visualize_yolo_results(image_path, result, output_path, show_plot=args.show_plots)
            
            # Save keypoints to JSON
            json_data = save_keypoints_json(image_path, result, json_path)
            
            # Count results
            objects_count = len(json_data["objects"])
            keypoints_count = sum(len(obj["keypoints"]) for obj in json_data["objects"])
            
            total_processed += 1
            total_objects += objects_count
            total_keypoints += keypoints_count
            
            print(f"  ✓ Objects: {objects_count}, Keypoints: {keypoints_count}")
            print(f"  ✓ Image saved: {output_path}")
            print(f"  ✓ JSON saved: {json_path}")
                
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
    
    # Print final summary
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Images processed: {total_processed}/{len(image_files)}")
    print(f"Total objects detected: {total_objects}")
    print(f"Total keypoints detected: {total_keypoints}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"  - PNG files: result_*.png")
    print(f"  - JSON files: keypoints_*.json")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()