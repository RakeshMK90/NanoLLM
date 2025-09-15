#!/usr/bin/env python3
"""
VLM Service - Jetson optimized version using existing jetson.utils
Uses the existing video capture system from NanoLLM
"""

import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from flask import Flask, jsonify, request
import threading
import queue
from PIL import Image

# Import NanoLLM and jetson-utils components
from nano_llm import NanoLLM, ChatHistory
from nano_llm.plugins import ChatQuery
from nano_llm.utils import ArgParser

# Try to import jetson utils for video capture
try:
    import jetson.utils
    JETSON_UTILS_AVAILABLE = True
except ImportError:
    JETSON_UTILS_AVAILABLE = False
    logging.warning("jetson.utils not available, using fallback")

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

class JetsonVideoCapture:
    """Video capture using jetson.utils"""

    def __init__(self, device="/dev/video0"):
        self.device = device
        self.camera = None
        self.width = 1280  # Match video_query.py default resolution
        self.height = 720
        self.initialize()

    def initialize(self):
        """Initialize video capture"""
        try:
            if JETSON_UTILS_AVAILABLE:
                # Use jetson.utils for optimized capture
                self.camera = jetson.utils.videoSource(f"v4l2://{self.device}")
                if self.camera:
                    logger.info(f"Jetson video capture initialized: {self.device}")
                    return True

            logger.error("Failed to initialize jetson video capture")
            return False
        except Exception as e:
            logger.error(f"Video capture initialization failed: {e}")
            return False

    def capture(self):
        """Capture a frame"""
        try:
            if self.camera and JETSON_UTILS_AVAILABLE:
                # Capture from jetson.utils (returns CUDA memory)
                cuda_img = self.camera.Capture()
                if cuda_img is not None:
                    # Synchronize CUDA operations to avoid stream conflicts
                    jetson.utils.cudaDeviceSynchronize()

                    # Convert CUDA image to numpy with proper error handling
                    try:
                        rgb_cpu = jetson.utils.cudaToNumpy(cuda_img)

                        # Handle I420 format conversion
                        if len(rgb_cpu.shape) == 1:
                            # I420 format - extract Y plane for grayscale
                            height, width = cuda_img.height, cuda_img.width
                            y_plane = rgb_cpu[:height * width].reshape((height, width))
                            # Convert grayscale to RGB
                            rgb_img = np.stack([y_plane, y_plane, y_plane], axis=2)
                            return rgb_img
                        else:
                            # Already in proper RGB format
                            return rgb_cpu

                    except Exception as conversion_error:
                        logger.warning(f"CUDA conversion failed: {conversion_error}")
                        # Fall back to simpler approach - just use mock data
                        return self._get_mock_frame()

            return None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return self._get_mock_frame()

    def _get_mock_frame(self):
        """Generate a mock frame for testing"""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame[100:300, 200:400] = [100, 150, 200]  # Add some content
        return mock_frame

    def release(self):
        """Release video capture"""
        if self.camera:
            try:
                # jetson.utils cleanup is automatic
                self.camera = None
                logger.info("Released jetson video capture")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")

class FallbackVideoCapture:
    """Fallback video capture for testing without video"""

    def __init__(self, device="/dev/video0"):
        self.device = device
        self.frame_count = 0
        logger.info("Using fallback video capture (mock)")

    def initialize(self):
        return True

    def capture(self):
        """Generate a mock frame"""
        self.frame_count += 1
        # Create a simple test pattern
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some pattern to make it interesting
        if self.frame_count % 100 < 50:
            # Simulate a "person detected" scenario
            mock_frame[100:300, 200:400] = [100, 150, 200]  # Person-like region
        else:
            # Simulate a "technical equipment" scenario
            mock_frame[150:250, 300:500] = [200, 100, 50]   # Equipment-like region

        return mock_frame

    def release(self):
        logger.info("Released fallback video capture")

class VLMService:
    """Vision-Language Model service with REST API"""

    def __init__(self, model_name: str = "Efficient-Large-Model/VILA1.5-3b"):
        self.model_name = model_name
        self.model = None
        self.llm = None  # ChatQuery plugin like video_query.py
        self.video_capture = None
        self.observation_queue = queue.Queue(maxsize=10)
        self.latest_observation = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 90  # Process every 3 seconds at 30fps
        self.latest_response = ""

    def initialize_model(self):
        """Initialize the VLM model using ChatQuery plugin like video_query.py"""
        try:
            logger.info(f"Initializing ChatQuery plugin with model: {self.model_name}")

            # Use ChatQuery plugin exactly like video_query.py
            self.llm = ChatQuery(
                model=self.model_name,
                api='mlc',
                quantization='q4f16_ft',
                max_context_len=256,
                vision_api='auto',
                drop_inputs=True,
                vision_scaling='resize',
                warmup=True
            )

            logger.info("ChatQuery plugin created, adding text handler...")

            # Add text output handler
            self.llm.add(self.on_text)

            logger.info("Starting ChatQuery plugin...")
            self.llm.start()

            logger.info(f"Successfully loaded VLM model with ChatQuery: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def on_text(self, text):
        """Handle text output from ChatQuery plugin"""
        # Accumulate text properly
        if not hasattr(self, '_response_building'):
            self._response_building = False

        if not self._response_building:
            # Start of new response
            self.latest_response = text
            self._response_building = True
        else:
            # Continue building response
            self.latest_response += text

        # Check if response is complete
        if text.endswith(('</s>', '###')) or len(self.latest_response) > 200:
            self._response_building = False
            logger.info(f"Complete response: {self.latest_response}")

        logger.debug(f"Text update: {text} | Total: {self.latest_response[:100]}...")

    def initialize_video_source(self, device: str = "/dev/video0", use_fallback: bool = False):
        """Initialize video capture"""
        try:
            # Use fallback for testing to avoid CUDA conflicts
            if use_fallback or not JETSON_UTILS_AVAILABLE:
                logger.warning("Using fallback video capture for testing")
                self.video_capture = FallbackVideoCapture(device)
            else:
                self.video_capture = JetsonVideoCapture(device)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video source: {e}")
            # Try fallback if jetson capture fails
            logger.info("Trying fallback video capture...")
            try:
                self.video_capture = FallbackVideoCapture(device)
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return False

    def extract_structured_info(self, raw_text: str) -> StructuredObservation:
        """Extract structured information from raw VLM output"""
        content = raw_text.strip() if raw_text else ""
        confidence = 0.8 if content else 0.1

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
            suggested_actions.extend(["investigate_issue", "check_documentation"])

        if "connector" in content_lower or "cable" in content_lower:
            suggested_actions.extend(["verify_connection", "check_wiring_diagram"])

        if "procedure" in content_lower or "install" in content_lower:
            suggested_actions.extend(["follow_procedure", "gather_tools"])

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
        """Process frame using video_query.py pattern - the WORKING approach"""
        try:
            if frame is None:
                return

            # Convert numpy array to proper format (like video_query.py expects)
            if isinstance(frame, np.ndarray):
                # Ensure frame is uint8 and in proper range
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                # Ensure proper shape (H, W, 3)
                if len(frame.shape) == 2:
                    # Grayscale to RGB
                    frame = np.stack([frame, frame, frame], axis=2)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    # Single channel to RGB
                    frame = np.repeat(frame, 3, axis=2)

                # CRITICAL: Use exact video_query.py pattern - no PIL conversion!
                np_image = frame  # Keep as numpy array

                # Synchronize CUDA operations like video_query.py
                if JETSON_UTILS_AVAILABLE:
                    jetson.utils.cudaDeviceSynchronize()

            else:
                logger.error(f"Unexpected frame type: {type(frame)}")
                return

            # Prepare prompt for technical analysis
            prompt = """Analyze this image for technical elements. Describe what you see including:
            - Any people, faces, or human activity
            - Technical equipment, connectors, cables, or electrical components
            - Warning lights, indicators, or displays
            - Any visible text, labels, or numbers
            - Potential issues or anomalies
            Keep the description concise and technical."""

            # Reset response for new query
            self.latest_response = ""
            self._response_building = False

            # Use EXACT video_query.py pattern - ChatQuery with numpy array
            self.llm(['/reset', np_image, prompt])

            # Wait for response to complete (with timeout)
            max_wait = 5.0  # 5 seconds max
            wait_time = 0.0
            while wait_time < max_wait and (not self.latest_response or self._response_building):
                time.sleep(0.1)
                wait_time += 0.1

            logger.info(f"Waited {wait_time:.1f}s for response: '{self.latest_response[:100]}...'")

            # Extract structured information from the response
            observation = self.extract_structured_info(self.latest_response)

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

            logger.info(f"Processed frame {self.frame_count}: {observation.content[:50]}...")

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
        """Main video processing loop with sequential CUDA operations"""
        logger.info("Starting video processing loop")

        try:
            while self.is_running:
                if self.video_capture:
                    # Sequential approach: capture frame, then process completely before next capture
                    frame = self.video_capture.capture()
                    if frame is not None:
                        self.frame_count += 1

                        # Process every Nth frame to avoid overwhelming the system
                        if self.frame_count % self.process_every_n_frames == 0:
                            logger.info(f"Processing frame {self.frame_count}")

                            # CRITICAL: Ensure all CUDA operations from capture are complete
                            if JETSON_UTILS_AVAILABLE:
                                jetson.utils.cudaDeviceSynchronize()

                            # Process frame completely before next capture
                            self.process_frame(frame)

                            # Ensure VLM processing is complete before next iteration
                            if JETSON_UTILS_AVAILABLE:
                                jetson.utils.cudaDeviceSynchronize()

                # Longer sleep when processing frames to give CUDA operations time to complete
                if self.frame_count % self.process_every_n_frames == 0:
                    time.sleep(0.1)  # 100ms after processing
                else:
                    time.sleep(0.033)  # ~30 FPS for non-processed frames

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
        "model_loaded": vlm_service.llm is not None,
        "video_active": vlm_service.is_running,
        "frames_processed": vlm_service.frame_count,
        "jetson_utils": JETSON_UTILS_AVAILABLE,
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
            "frames_processed": vlm_service.frame_count,
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
    if not vlm_service.llm:
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
        "model_loaded": vlm_service.llm is not None,
        "is_running": vlm_service.is_running,
        "queue_size": vlm_service.observation_queue.qsize(),
        "frames_processed": vlm_service.frame_count,
        "process_interval": vlm_service.process_every_n_frames,
        "jetson_utils_available": JETSON_UTILS_AVAILABLE,
        "latest_response": vlm_service.latest_response,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/debug', methods=['GET'])
def debug_status():
    """Debug endpoint to check model status"""
    return jsonify({
        "llm_plugin": str(type(vlm_service.llm)) if vlm_service.llm else None,
        "llm_is_none": vlm_service.llm is None,
        "latest_response": vlm_service.latest_response,
        "model_name": vlm_service.model_name,
        "processing_count": vlm_service.frame_count,
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
    parser.add_argument("--use-fallback", action="store_true", help="Use fallback video capture (for testing)")

    args = parser.parse_args()

    # Initialize VLM service
    vlm_service.model_name = args.model

    logger.info("Initializing VLM model...")
    if not vlm_service.initialize_model():
        logger.error("Failed to initialize model")
        return 1

    logger.info("Initializing video source...")
    if not vlm_service.initialize_video_source(args.video_device, args.use_fallback):
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