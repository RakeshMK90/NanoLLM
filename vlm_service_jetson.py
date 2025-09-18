#!/usr/bin/env python3
"""
VLM Service - Jetson optimized version using existing jetson.utils
Uses the existing video capture system from NanoLLM
"""

import os
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
from nano_llm import NanoLLM, ChatHistory, Agent
from nano_llm.plugins import ChatQuery, VideoSource, VideoOutput, PrintStream
from nano_llm.web import WebServer
from nano_llm.utils import ArgParser, wrap_text

# Try to import jetson utils for video capture
try:
    import jetson.utils
    from jetson_utils import cudaFont, cudaMemcpy, cudaToNumpy, cudaDeviceSynchronize, saveImage
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

class VLMService(Agent):
    """Vision-Language Model service with REST API and video output"""

    def __init__(self, model_name: str = "Efficient-Large-Model/VILA1.5-3b", **kwargs):
        super().__init__()

        self.model_name = model_name
        self.model = None
        self.llm = None  # ChatQuery plugin like video_query.py
        self.observation_queue = queue.Queue(maxsize=10)
        self.latest_observation = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 90  # Process every 3 seconds at 30fps
        self.latest_response = ""

        # Video processing state
        self.text = ""
        self.eos = False
        self.font = None
        self.last_image = None
        self.analyze_requested = False
        self.rag_service_url = kwargs.get('rag_service_url', 'http://localhost:8555')

        # Video streams (like video_query.py)
        if JETSON_UTILS_AVAILABLE:
            # Extract video parameters from kwargs
            video_input = kwargs.get('video_input', '/dev/video0')
            video_output = kwargs.get('video_output', 'webrtc://@:8554/output')

            self.video_source = VideoSource(video_input=video_input, cuda_stream=0)
            self.video_output = VideoOutput(video_output=video_output, cuda_stream=0)
            self.font = cudaFont()

            # Connect video processing chain (like video_query.py)
            self.video_source.add(self.on_video, threaded=False)
            self.video_source.add(self.video_output)  # Direct connection for display
            self.video_output.start()

            logger.info(f"Video source initialized: {video_input}")
            logger.info(f"Video output initialized: {video_output}")
        else:
            self.video_source = None
            self.video_output = None

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
            self.llm.add(PrintStream(color='green', relay=True).add(self.on_text))

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
        # Handle streaming text like video_query.py
        from nano_llm import StopTokens

        if self.eos:
            self.text = text  # reset rolling text
            self.eos = False  # new query response
        else:
            self.text = self.text + text

        if text.endswith(tuple(StopTokens + ['###', '</s>'])):
            self.eos = True
            # Process completed observation
            self.process_completed_observation()

        logger.debug(f"Text update: {text} | Current: {self.text[:100]}...")

    def process_completed_observation(self):
        """Process the completed VLM observation"""
        try:
            # Extract structured information
            observation = self.extract_structured_info(self.text)

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

            logger.info(f"Processed observation: {observation.content[:50]}...")

        except Exception as e:
            logger.error(f"Error processing observation: {e}")

    def on_video(self, image):
        """Process video frames with object overlay and RAG integration"""
        if not JETSON_UTILS_AVAILABLE or not self.font:
            return

        # Store last image for RAG analysis
        self.last_image = cudaMemcpy(image)

        # Process frame for VLM analysis (every N frames)
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames == 0:
            self.process_video_frame(image)

        # Draw overlays on the video
        self.draw_overlays(image)

        # Note: video output is handled automatically via direct connection
        # self.video_source.add(self.video_output) in constructor

    def process_video_frame(self, image):
        """Process video frame for VLM analysis"""
        try:
            np_image = cudaToNumpy(image)
            cudaDeviceSynchronize()

            prompt = """Analyze this image for technical elements. List any detected objects, equipment, or issues. Be concise."""

            self.llm(['/reset', np_image, prompt])

        except Exception as e:
            logger.error(f"Error processing video frame: {e}")

    def draw_overlays(self, image):
        """Draw text overlays and object tags on video"""
        if not self.font:
            return

        y = 5

        # Draw latest VLM analysis
        if self.text:
            clean_text = self.text.replace('\n', '').replace('</s>', '').strip()
            y = wrap_text(self.font, image, text=f"Analysis: {clean_text}",
                         x=5, y=y, color=self.font.White, background=self.font.Gray40)

        # Draw detected objects with tags
        if self.latest_observation and self.latest_observation.detected_objects:
            objects_text = "Objects: " + ", ".join(self.latest_observation.detected_objects)
            y = wrap_text(self.font, image, text=objects_text,
                         x=5, y=y, color=(120,215,21), background=self.font.Gray40)

        # Draw analyze button prompt
        y = wrap_text(self.font, image, text="Press 'A' for RAG Analysis",
                     x=5, y=y, color=(255,172,28), background=self.font.Gray40)

        # Draw frame counter
        y = wrap_text(self.font, image, text=f"Frame: {self.frame_count}",
                     x=5, y=y, color=(128,128,128), background=self.font.Gray40)

    def setup_keyboard_handler(self):
        """Setup keyboard handler for analyze button"""
        def keyboard_thread():
            while self.is_running:
                try:
                    key = input().strip().lower()
                    if key == 'a' and self.latest_observation:
                        self.trigger_rag_analysis()
                except Exception as e:
                    continue

        thread = threading.Thread(target=keyboard_thread, daemon=True)
        thread.start()

    def trigger_rag_analysis(self):
        """Trigger RAG analysis of current observation"""
        if not self.latest_observation or not self.latest_observation.detected_objects:
            logger.warning("No objects detected for RAG analysis")
            return

        try:
            # Build query from detected objects and observation
            objects = ", ".join(self.latest_observation.detected_objects)
            query = f"Troubleshooting for {objects}: {self.latest_observation.content[:100]}"

            logger.info(f"Triggering RAG analysis: {query}")

            # Call RAG service
            response = self.call_rag_service(query)

            if response:
                logger.info(f"RAG response: {response[:200]}...")
                # You could save this to a file, display in UI, etc.
                self.save_rag_response(query, response)

        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}")

    def call_rag_service(self, query: str) -> str:
        """Call the RAG+LLM service"""
        try:
            import requests

            url = f"{self.rag_service_url}/query"
            data = {
                "query": query,
                "k": 1
            }

            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get('response', '')

        except Exception as e:
            logger.error(f"Failed to call RAG service: {e}")
            return ""

    def save_rag_response(self, query: str, response: str):
        """Save RAG response to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/data/rag_responses/response_{timestamp}.txt"

            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'w') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write(f"Timestamp: {timestamp}\n")

            logger.info(f"Saved RAG response to {filename}")

        except Exception as e:
            logger.error(f"Failed to save RAG response: {e}")

    def setup_webserver(self, **kwargs):
        """Setup WebServer for video output"""
        if not JETSON_UTILS_AVAILABLE:
            return

        try:
            # Setup WebRTC video streaming
            video_source = self.video_source.stream.GetOptions()['resource']
            video_output = self.video_output.stream.GetOptions()['resource']

            webrtc_args = {}

            if video_source['protocol'] == 'webrtc':
                webrtc_args.update(dict(webrtc_input_stream=video_source['path'].strip('/'),
                                       webrtc_input_port=video_source['port'],
                                       send_webrtc=True))
            else:
                webrtc_args.update(dict(webrtc_input_stream='input',
                                       webrtc_input_port=8554,
                                       send_webrtc=False))

            if video_output['protocol'] == 'webrtc':
                webrtc_args.update(dict(webrtc_output_stream=video_output['path'].strip('/'),
                                       webrtc_output_port=video_output['port']))
            else:
                webrtc_args.update(dict(webrtc_output_stream='output',
                                       webrtc_output_port=8554))

            # Create web server for video streaming
            self.server = WebServer(
                msg_callback=self.on_websocket,
                index='video_query.html',
                title='VLM Technical Analysis',
                model=os.path.basename(self.model_name),
                **webrtc_args,
                **kwargs
            )

            logger.info("WebServer setup for video streaming")

        except Exception as e:
            logger.error(f"Failed to setup WebServer: {e}")

    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        """Handle WebSocket messages"""
        if msg_type == WebServer.MESSAGE_JSON:
            if 'analyze' in msg:
                # Trigger RAG analysis via websocket
                self.trigger_rag_analysis()
            elif 'rag_service_url' in msg:
                self.rag_service_url = msg['rag_service_url']
                logger.info(f"Updated RAG service URL: {self.rag_service_url}")

    def start_video_processing(self):
        """Start video processing and web server"""
        if not JETSON_UTILS_AVAILABLE:
            logger.warning("Video processing not available without jetson.utils")
            return

        try:
            # Setup keyboard handler
            self.setup_keyboard_handler()

            # Start video source (this is critical for display output!)
            if self.video_source:
                logger.info("Starting video source...")
                self.video_source.start()

            # Start web server for video streaming
            if hasattr(self, 'server'):
                self.server.start()

            logger.info("Video processing started with web interface")

        except Exception as e:
            logger.error(f"Failed to start video processing: {e}")

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
            "phone", "mug", "cup", "wire","connector", "cable", "wire", "button", "switch", "panel", "display",
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
    parser.add_argument("--enable-video-output", action="store_true", help="Enable video output with overlays")
    parser.add_argument("--rag-service-url", default="http://localhost:8555", help="RAG service URL")
    parser.add_argument("--video-input", default="/dev/video0", help="Video input device")
    parser.add_argument("--video-output", default="webrtc://@:8554/output", help="Video output stream")

    args = parser.parse_args()

    # Initialize VLM service with video options
    vlm_service = VLMService(
        model_name=args.model,
        rag_service_url=args.rag_service_url,
        video_input=args.video_input,
        video_output=args.video_output
    )

    logger.info("Initializing VLM model...")
    if not vlm_service.initialize_model():
        logger.error("Failed to initialize model")
        return 1

    # Setup video output if enabled
    if args.enable_video_output:
        logger.info("Setting up video output...")
        vlm_service.setup_webserver()

    # Initialize video source only if not using native video output
    if not args.enable_video_output:
        logger.info("Initializing video source...")
        if not vlm_service.initialize_video_source(args.video_device, args.use_fallback):
            logger.error("Failed to initialize video source")
            return 1
    else:
        logger.info("Using native VideoSource/VideoOutput for display output")

    # Auto-start if requested
    if args.auto_start:
        if args.enable_video_output:
            # Use native video processing with VideoSource/VideoOutput plugins
            logger.info("Starting native video processing with display output...")
            vlm_service.start_video_processing()
        else:
            # Use fallback video processing for API-only mode
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