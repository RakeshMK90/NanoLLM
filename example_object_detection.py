#!/usr/bin/env python3
"""
Example script demonstrating object detection with customizable object lists.

This script shows how to use the ObjectDetection agent to:
1. Define a list of objects to detect
2. Set detection thresholds
3. View real-time object detection with bounding boxes
4. Use the web interface for interactive control

Usage:
    python3 example_object_detection.py --video-input /dev/video0 --target-objects person car bicycle dog
"""

import sys
import os

# Add the nano_llm directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nano_llm'))

from nano_llm.agents.object_detection import ObjectDetection
from nano_llm.utils import ArgParser

def main():
    # Parse command line arguments
    parser = ArgParser(extras=['video_input', 'video_output', 'web', 'nanodb'])
    parser.add_argument("--target-objects", nargs='+', 
                       default=['person', 'car', 'bicycle', 'dog', 'cat', 'truck', 'bus', 'motorcycle'],
                       help="List of objects to detect and highlight")
    parser.add_argument("--detection-threshold", type=float, default=0.5,
                       help="Confidence threshold for object detection (0.0 to 1.0)")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-13b",
                       help="Vision-language model to use for object detection")
    parser.add_argument("--web-title", type=str, default="Object Detection Demo",
                       help="Title for the web interface")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OBJECT DETECTION DEMO")
    print("=" * 60)
    print(f"Target Objects: {', '.join(args.target_objects)}")
    print(f"Detection Threshold: {args.detection_threshold}")
    print(f"Model: {args.model}")
    print(f"Video Input: {args.video_input}")
    print(f"Video Output: {args.video_output}")
    print("=" * 60)
    print()
    print("Starting object detection agent...")
    print("Web interface will be available at: https://localhost:8050")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Create and run the object detection agent
        agent = ObjectDetection(
            model=args.model,
            target_objects=args.target_objects,
            detection_threshold=args.detection_threshold,
            web_title=args.web_title,
            **vars(args)
        )
        
        # Start the agent
        agent.run()
        
    except KeyboardInterrupt:
        print("\nStopping object detection agent...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
