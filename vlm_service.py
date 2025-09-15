#!/usr/bin/env python3
"""
VLM Service - Wraps nano_llm video_query agent with REST API
Provides structured observations from live video stream
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from flask import Flask, jsonify, request
import threading
import queue

from PIL import Image
from nano_llm import NanoLLM
from nano_llm.utils import ArgParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StructuredObservation:
    """Structured observation output"""
    timestamp: str
    content: str
    confidence: float
    detected_objects: List[str]
    suggested_actions: List[str]
    raw_output: str

class SimpleVideoCapture:
    """Simple video capture wrapper"""

    def __init__(self, device="/dev/video0"):
        self.device = device
        self.cap = None
        self.initialize()

    def initialize(self):
        """Initialize video capture"""
        try:
            # Try OpenCV first
            self.cap = cv2.VideoCapture(self.device)
            if not self.cap.isOpened():
                # Try with different backend
                self.cap = cv2.VideoCapture(0)  # Use default camera

            if self.cap.isOpened():
                # Set properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                logger.info("Video capture initialized successfully")
                return True
            else:
                logger.error("Failed to open video capture")
                return False
        except Exception as e:
            logger.error(f"Video capture initialization failed: {e}")
            return False

    def capture(self):
        """Capture a frame"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()

class VLMService:
    """Vision-Language Model service with REST API"""

    def __init__(self, model_name: str = "Efficient-Large-Model/VILA1.5-3b"):
        self.model_name = model_name
        self.model = None
        self.video_capture = None
        self.observation_queue = queue.Queue(maxsize=10)
        self.latest_observation = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 90  # Process every 3 seconds at 30fps

    def initialize_model(self):
        """Initialize the VLM model"""
        try:
            self.model = NanoLLM.from_pretrained(
                self.model_name,
                api='mlc',
                quantization='q4f16_ft',
                max_context_len=256,
                vision_api='auto'
            )
            logger.info(f"Loaded VLM model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def initialize_video_source(self, device: str = "/dev/video0"):
        """Initialize video capture"""
        try:
            self.video_capture = SimpleVideoCapture(device)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video source: {e}")
            return False

    def extract_structured_info(self, raw_text: str) -> StructuredObservation:
        """Extract structured information from raw VLM output"""
        # Parse the raw output to extract structured information
        content = raw_text.strip()
        confidence = 0.8  # Default confidence

        # Extract detected objects (simple keyword matching)
        detected_objects = []
        object_keywords = [
            "person", "man", "woman", "face", "hand", "phone", "mug", "cup",
            "connector", "cable", "wire", "button", "switch", "panel", "display",
            "warning", "light", "indicator", "gauge", "meter", "sensor",
            "screw", "bolt", "nut", "cover", "housing", "bracket"
        ]

        content_lower = content.lower()
        for keyword in object_keywords:
            if keyword in content_lower:
                detected_objects.append(keyword)

        # Extract suggested actions based on content
        suggested_actions = []
        if "warning" in content_lower or "error" in content_lower:
            suggested_actions.append("investigate_issue")
            suggested_actions.append("check_documentation")

        if "connector" in content_lower or "cable" in content_lower:
            suggested_actions.append("verify_connection")
            suggested_actions.append("check_wiring_diagram")

        if "procedure" in content_lower or "install" in content_lower:
            suggested_actions.append("follow_procedure")
            suggested_actions.append("gather_tools")

        if "person" in content_lower or "face" in content_lower:
            suggested_actions.extend(["continue_monitoring", "analyze_behavior"])

        # Default actions if none found
        if not suggested_actions:
            suggested_actions = ["continue_monitoring"]

        # Estimate confidence based on output length and clarity
        if len(content) > 50 and any(obj in content_lower for obj in object_keywords):
            confidence = 0.9
        elif len(content) > 20:
            confidence = 0.7
        elif len(content) > 5:
            confidence = 0.5
        else:
            confidence = 0.1

        return StructuredObservation(
            timestamp=datetime.now().isoformat(),
            content=content,
            confidence=confidence,
            detected_objects=detected_objects,
            suggested_actions=suggested_actions,
            raw_output=raw_text
        )

    def process_frame(self, frame):
        """Process a single video frame with VLM"""
        try:
            if frame is None:
                return

            # Convert OpenCV BGR to RGB
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Prepare prompt for technical analysis
            prompt = """Analyze this image for technical elements. Describe what you see including:
            - Any people, faces, or human activity
            - Technical equipment, connectors, cables, or electrical components
            - Warning lights, indicators, or displays
            - Any visible text, labels, or numbers
            - Potential issues or anomalies
            Keep the description concise and technical."""

            # Generate response from VLM
            response = self.model.generate(
                pil_image,
                prompt=prompt,
                max_new_tokens=32,
                temperature=0.1
            )

            # Get the response text
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, '__iter__'):
                # If it's a generator, collect the tokens
                response_text = ''.join(str(token) for token in response)
            else:
                response_text = str(response)

            # Extract structured information
            observation = self.extract_structured_info(response_text)

            # Update latest observation
            self.latest_observation = observation

            # Add to queue (non-blocking)
            try:
                self.observation_queue.put_nowait(observation)
            except queue.Full:
                # Remove oldest observation if queue is full
                try:
                    self.observation_queue.get_nowait()
                    self.observation_queue.put_nowait(observation)
                except queue.Empty:
                    pass

            logger.info(f"Processed frame: {observation.content[:50]}...")

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Create an error observation
            error_observation = StructuredObservation(
                timestamp=datetime.now().isoformat(),
                content=f"Processing error: {str(e)[:100]}",
                confidence=0.0,
                detected_objects=["error"],
                suggested_actions=["check_system", "restart_processing"],
                raw_output=str(e)
            )
            self.latest_observation = error_observation

    def video_processing_loop(self):
        """Main video processing loop"""
        logger.info("Starting video processing loop")

        try:
            while self.is_running:
                if self.video_capture:
                    frame = self.video_capture.capture()
                    if frame is not None:
                        self.frame_count += 1

                        # Process every Nth frame to avoid overwhelming the system
                        if self.frame_count % self.process_every_n_frames == 0:
                            logger.info(f"Processing frame {self.frame_count}")
                            self.process_frame(frame)

                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logger.error(f"Error in video processing loop: {e}")
        finally:
            logger.info("Video processing loop ended")

    def start_processing(self):
        """Start the video processing in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self.video_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started video processing thread")

    def stop_processing(self):
        """Stop the video processing"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5)
        if self.video_capture:
            self.video_capture.release()
        logger.info("Stopped video processing")

# Flask REST API
app = Flask(__name__)
vlm_service = VLMService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": vlm_service.model is not None,
        "video_active": vlm_service.is_running,
        "frames_processed": vlm_service.frame_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/observation', methods=['GET'])
def get_latest_observation():
    """Get the latest structured observation"""
    if vlm_service.latest_observation:
        return jsonify(asdict(vlm_service.latest_observation))
    else:
        return jsonify({
            "error": "No observations available yet",
            "message": "Wait for video processing to generate observations",
            "timestamp": datetime.now().isoformat()
        }), 404

@app.route('/observations/history', methods=['GET'])
def get_observation_history():
    """Get recent observation history"""
    observations = []
    temp_queue = queue.Queue()

    # Extract all observations from queue
    while not vlm_service.observation_queue.empty():
        try:
            obs = vlm_service.observation_queue.get_nowait()
            observations.append(asdict(obs))
            temp_queue.put(obs)
        except queue.Empty:
            break

    # Put observations back
    while not temp_queue.empty():
        vlm_service.observation_queue.put(temp_queue.get())

    return jsonify({
        "observations": observations,
        "count": len(observations),
        "frames_processed": vlm_service.frame_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/start', methods=['POST'])
def start_processing():
    """Start video processing"""
    if not vlm_service.model:
        return jsonify({"error": "Model not initialized"}), 500

    vlm_service.start_processing()
    return jsonify({
        "status": "started",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stop', methods=['POST'])
def stop_processing():
    """Stop video processing"""
    vlm_service.stop_processing()
    return jsonify({
        "status": "stopped",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        "model_name": vlm_service.model_name,
        "is_running": vlm_service.is_running,
        "queue_size": vlm_service.observation_queue.qsize(),
        "frames_processed": vlm_service.frame_count,
        "process_interval": vlm_service.process_every_n_frames,
        "timestamp": datetime.now().isoformat()
    })

def main():
    """Main function to initialize and run the VLM service"""
    import argparse

    parser = argparse.ArgumentParser(description="VLM Service with REST API")
    parser.add_argument("--model", default="Efficient-Large-Model/VILA1.5-3b", help="VLM model name")
    parser.add_argument("--video-device", default="/dev/video0", help="Video device path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8554, help="Port to bind to")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start video processing")

    args = parser.parse_args()

    # Initialize VLM service
    vlm_service.model_name = args.model

    logger.info("Initializing VLM model...")
    if not vlm_service.initialize_model():
        logger.error("Failed to initialize model")
        return 1

    logger.info("Initializing video source...")
    if not vlm_service.initialize_video_source(args.video_device):
        logger.error("Failed to initialize video source")
        return 1

    # Auto-start if requested
    if args.auto_start:
        vlm_service.start_processing()

    # Start Flask server
    logger.info(f"Starting VLM service on {args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down VLM service")
    finally:
        vlm_service.stop_processing()

    return 0

if __name__ == "__main__":
    exit(main())