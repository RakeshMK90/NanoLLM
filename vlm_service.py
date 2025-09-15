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

from nano_llm import NanoLLM
from nano_llm.plugins import VideoSource, VideoOutput, TextOverlay
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

class VLMService:
    """Vision-Language Model service with REST API"""

    def __init__(self, model_name: str = "Efficient-Large-Model/VILA1.5-3b"):
        self.model_name = model_name
        self.model = None
        self.video_source = None
        self.observation_queue = queue.Queue(maxsize=10)
        self.latest_observation = None
        self.is_running = False

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
            self.video_source = VideoSource(device)
            logger.info(f"Initialized video source: {device}")
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

        # Estimate confidence based on output length and clarity
        if len(content) > 50 and any(obj in content_lower for obj in object_keywords):
            confidence = 0.9
        elif len(content) > 20:
            confidence = 0.7
        else:
            confidence = 0.5

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
            # Prepare prompt for technical analysis
            prompt = """Analyze this image for technical elements. Describe what you see including:
            - Any connectors, cables, or electrical components
            - Warning lights, indicators, or displays
            - Technical panels, buttons, or controls
            - Any visible text, labels, or numbers
            - Potential issues or anomalies
            Keep the description concise and technical."""

            # Generate response from VLM
            response = self.model.generate(
                frame,
                prompt=prompt,
                max_new_tokens=32,
                temperature=0.1
            )

            # Extract structured information
            observation = self.extract_structured_info(response.text)

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

            logger.debug(f"Processed frame: {observation.content[:50]}...")

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def video_processing_loop(self):
        """Main video processing loop"""
        logger.info("Starting video processing loop")

        frame_count = 0
        process_every_n_frames = 30  # Process every 30 frames (1 second at 30fps)

        try:
            while self.is_running:
                frame = self.video_source.capture()
                if frame is not None:
                    frame_count += 1

                    # Process every Nth frame to avoid overwhelming the system
                    if frame_count % process_every_n_frames == 0:
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
        "timestamp": datetime.now().isoformat()
    })

@app.route('/observation', methods=['GET'])
def get_latest_observation():
    """Get the latest structured observation"""
    if vlm_service.latest_observation:
        return jsonify(asdict(vlm_service.latest_observation))
    else:
        return jsonify({
            "error": "No observations available",
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