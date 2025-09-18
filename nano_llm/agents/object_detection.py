#!/usr/bin/env python3
import os
import time
import json
import torch
import pprint
import logging
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional

from datetime import datetime
from termcolor import cprint

from nano_llm import Agent, StopTokens
from nano_llm.web import WebServer
from nano_llm.plugins import VideoSource, VideoOutput, ChatQuery, PrintStream, ProcessProxy, EventFilter, NanoDB
from nano_llm.utils import ArgParser, print_table, wrap_text

from jetson_utils import cudaFont, cudaMemcpy, cudaToNumpy, cudaDeviceSynchronize, saveImage


class ObjectDetection(Agent):
    """
    Object detection agent that can identify and highlight specific objects in video frames.
    Extends VideoQuery with object detection capabilities using vision-language models.
    """
    
    def __init__(self, model="liuhaotian/llava-v1.5-13b", nanodb=None, vision_scaling='resize', 
                 target_objects=None, detection_threshold=0.5, **kwargs):
        """
        Args:
            model (NanoLLM|str): the NanoLLM multimodal model instance, or name/path of a multimodal model to load.
            nanodb (NanoDB|str): optional NanoDB plugin instance (or path to a NanoDB on disk) to match the incoming stream against.
            vision_scaling (str): ``'resize'`` to ignore aspect ratio when downscaling to the often-square resolution of the vision encoder,
                                  or ``crop`` to center-crop the images first to maintain aspect ratio (while discarding pixels).
            target_objects (List[str]): List of objects to detect and highlight (e.g., ['person', 'car', 'dog'])
            detection_threshold (float): Confidence threshold for object detection (0.0 to 1.0)
            kwargs: forwarded to the plugin initializers for ChatQuery, VideoSource, and VideoOutput
        """
        super().__init__()

        if not vision_scaling:
            vision_scaling = 'resize'
            
        # Default target objects if none specified
        if target_objects is None:
            target_objects = ['person', 'car', 'bicycle', 'dog', 'cat', 'truck', 'bus', 'motorcycle']
        
        self.target_objects = target_objects
        self.detection_threshold = detection_threshold
        self.detected_objects = []  # Current frame detected objects
        self.object_colors = self._generate_object_colors()
        
        #: The model plugin (ChatQuery)
        self.llm = ChatQuery(model=model, drop_inputs=True, vision_scaling=vision_scaling, warmup=True, **kwargs)
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_text))
        self.llm.start()

        self.text = ""
        self.eos = False

        # create video streams    
        self.video_source = VideoSource(**kwargs, cuda_stream=0)  #: The video source plugin
        self.video_output = VideoOutput(**kwargs, cuda_stream=0)  #: The video output plugin
        
        self.video_source.add(self.on_video, threaded=False)
        self.video_output.start()
        
        self.font = cudaFont()
        
        self.pause_video = False
        self.pause_image = None
        self.last_image = None
        self.tag_image = None

        self.pipeline = [self.video_source]
        
        # setup prompts for object detection
        self.prompt_history = kwargs.get('prompt')
        
        if not self.prompt_history:
            self.prompt_history = [
                f'Identify and locate all objects in this image. Look for: {", ".join(self.target_objects)}. For each detected object, provide the object name and approximate location (top-left, top-right, bottom-left, bottom-right, center).'
            ]
        
        self.prompt = self.prompt_history[0]
        
        self.last_prompt = None
        self.auto_refresh = True
        self.auto_refresh_db = True
        
        self.rag_threshold = 1.0
        self.rag_prompt = None
        self.rag_prompt_last = None
        
        self.keyboard_prompt = 0
        self.keyboard_thread = threading.Thread(target=self.poll_keyboard)
        self.keyboard_thread.start()

        # load vector database
        if nanodb:
            self.db_share_embed = False
            
            #: The `NanoDB <https://github.com/dusty-nv/jetson-containers/tree/master/packages/vectordb/nanodb>`_ vector database 
            self.db = NanoDB(
                path=nanodb, 
                model=None if self.db_share_embed else 'ViT-L/14@336px',
                reserve=kwargs.get('nanodb_reserve'), 
                top_k=18, drop_inputs=True,
            ).start().add(self.on_search)
            
            if self.db_share_embed:
                self.llm.add(self.on_image_embedding, channel=ChatQuery.OutputImageEmbedding)
        else:
            self.db = None
            
        # webserver
        mounts = {
            scan : f"/images/{n}" 
            for n, scan in enumerate(self.db.scans)
        } if self.db else {}
        
        mounts['/data/datasets/uploads'] = '/images/uploads'
        
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

        web_title = kwargs.get('web_title')
        web_title = web_title if web_title else 'OBJECT DETECTION'
        
        #: the webserver (by default on ``https://localhost:8050``)
        self.server = WebServer(
            msg_callback=self.on_websocket, 
            index='object_detection.html', 
            title=web_title, 
            model=os.path.basename(model),
            mounts=mounts,
            nanodb=nanodb,
            **webrtc_args,
            **kwargs
        )
        
        #: event filters for parsing bot output and triggering actions when conditions are met.
        self.events = EventFilter(server=self.server)
   
    def _generate_object_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate distinct colors for each target object type."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),    # Dark Green
        ]
        
        object_colors = {}
        for i, obj in enumerate(self.target_objects):
            object_colors[obj] = colors[i % len(colors)]
        
        return object_colors

    def parse_object_detections(self, text: str) -> List[Dict]:
        """
        Parse object detection results from model text output.
        Expected format: "person at center", "car at top-left", etc.
        """
        detections = []
        text_lower = text.lower()
        
        for obj in self.target_objects:
            if obj.lower() in text_lower:
                # Try to extract position information
                position = "center"  # default
                if "top-left" in text_lower:
                    position = "top-left"
                elif "top-right" in text_lower:
                    position = "top-right"
                elif "bottom-left" in text_lower:
                    position = "bottom-left"
                elif "bottom-right" in text_lower:
                    position = "bottom-right"
                elif "center" in text_lower:
                    position = "center"
                
                # Calculate bounding box based on position
                bbox = self._position_to_bbox(position)
                
                detections.append({
                    'object': obj,
                    'position': position,
                    'bbox': bbox,
                    'confidence': self.detection_threshold  # Default confidence
                })
        
        return detections

    def _position_to_bbox(self, position: str) -> Tuple[int, int, int, int]:
        """Convert position string to bounding box coordinates (x1, y1, x2, y2)."""
        # These are relative coordinates that will be scaled to actual image size
        position_map = {
            "top-left": (0.1, 0.1, 0.4, 0.4),
            "top-right": (0.6, 0.1, 0.9, 0.4),
            "bottom-left": (0.1, 0.6, 0.4, 0.9),
            "bottom-right": (0.6, 0.6, 0.9, 0.9),
            "center": (0.3, 0.3, 0.7, 0.7),
        }
        
        return position_map.get(position, (0.3, 0.3, 0.7, 0.7))

    def draw_bounding_boxes(self, image, detections: List[Dict]):
        """Draw bounding boxes and labels on the image using text overlays."""
        if not detections:
            return
            
        # For now, just display the detected objects as text overlays
        # This avoids dependency on potentially unavailable drawing functions
        detection_text = "Detected: " + ", ".join([f"{d['object']} ({d['confidence']:.2f})" for d in detections])
        
        # Draw detection summary at the top of the image
        wrap_text(self.font, image, text=detection_text, x=5, y=5, 
                 color=(255, 255, 0), background=self.font.Gray40)

    def on_video(self, image):
        """
        When a new frame is received from the video source, run the model on it with the set prompt,
        parse object detections, draw bounding boxes, and send it to the output video stream.
        """
        if self.pause_video:
            if not self.pause_image:
                self.pause_image = cudaMemcpy(image)
            image = cudaMemcpy(self.pause_image)
        
        if self.auto_refresh or self.prompt != self.last_prompt or self.rag_prompt != self.rag_prompt_last:
            np_image = cudaToNumpy(image)
            cudaDeviceSynchronize()
            
            if self.rag_prompt:
                prompt = self.rag_prompt + '. ' + self.prompt
            else:
                prompt = self.prompt
                
            self.llm(['/reset', np_image, prompt])
            
            self.last_prompt = self.prompt
            self.rag_prompt_last = self.rag_prompt
            
            if self.db:
                self.last_image = cudaMemcpy(image)

        # Draw bounding boxes for detected objects
        if self.detected_objects:
            self.draw_bounding_boxes(image, self.detected_objects)

        # draw text overlays
        text = self.text.replace('\n', '').replace('</s>', '').strip()
        y = 5
        
        if self.rag_prompt:
            y = wrap_text(self.font, image, text='RAG: ' + self.rag_prompt, x=5, y=y, color=(255,172,28), background=self.font.Gray40)
            
        y = wrap_text(self.font, image, text=self.prompt, x=5, y=y, color=(120,215,21), background=self.font.Gray40)

        if text:
            y = wrap_text(self.font, image, text=text, x=5, y=y, color=self.font.White, background=self.font.Gray40)
        
        # Display detection summary
        if self.detected_objects:
            detection_summary = f"Detected: {', '.join([d['object'] for d in self.detected_objects])}"
            y = wrap_text(self.font, image, text=detection_summary, x=5, y=y, color=(255, 255, 0), background=self.font.Gray40)
        
        self.video_output(image)
   
    def on_text(self, text):
        """
        When new output is received from the model, update the text to render,
        parse object detections, and check if it satisfied any of the event filters when the output is complete.
        """
        if self.eos:
            self.text = text  # reset rolling text
            self.eos = False  # new query response
        else:
            self.text = self.text + text

        if text.endswith(tuple(StopTokens + ['###'])):
            # Parse object detections from the complete text
            self.detected_objects = self.parse_object_detections(self.text)
            
            self.print_stats()

            self.events(self.text, prompt=self.prompt)
            
            if self.db and not self.db_share_embed:
                self.on_image_embedding(None) # use self.last_image instead of embeddings

            self.eos = True

    def on_image_embedding(self, embedding):
        """
        Receive the image embedding from CLIP that was used when the model processed
        the last image, and search it against the database to find the most similar
        images and their metadata for RAG.  Also, if the user requested the last image
        be tagged, add the embedding to the vector database along with the metadata tags.
        """
        if self.tag_image and self.last_image:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/data/datasets/uploads/{timestamp}.jpg"
            metadata = dict(path=filename, time=timestamp, tags=self.tag_image)
            self.tag_image = None 
            
            def save_image(filename, image, embedding, metadata):
                saveImage(filename, image)
                self.db(embedding if embedding is not None else image, add=True, metadata=metadata)
                logging.info(f"added incoming image to database with tags '{self.tag_image}' ({filename})")
        
            threading.Thread(target=save_image, args=(filename, self.last_image, embedding, metadata)).start()
            
        if self.auto_refresh_db:
            self.db(embedding if embedding is not None else self.last_image)
        
    def on_search(self, results):
        """
        Receive the similar matches from the vector database and update RAG with them,
        along with the most recent results shown in the web UI.
        """
        html = []
        
        for result in results:
            path = result['metadata']['path']
            for root, mount in self.server.mounts.items():
                if root in path:
                    html.append(dict(
                        image=path.replace(root, mount), 
                        similarity=f"{result['similarity']*100:.1f}%",
                        metadata=json.dumps(result['metadata'], indent=2).replace('"', '&quot;')
                    ))
                    
        if html:
            self.server.send_message({'search_results': html})
         
        if len(results) >= 3:
            cprint(f"nanodb search results (top 3)\n{pprint.pformat(results[:3], indent=2)}", color='blue')
        
        # RAG
        self.rag_prompt = None
        
        if len(results) == 0:    
            return
            
        result = results[0]

        if result['similarity'] > self.rag_threshold and 'tags' in result['metadata']:
            self.rag_prompt = f"This image is of {result['metadata']['tags']}"
        
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        """
        Websocket message handler from the client.
        """
        if msg_type == WebServer.MESSAGE_JSON:
            if 'prompt' in msg:
                self.prompt = msg['prompt']
                if self.prompt not in self.prompt_history:
                    self.prompt_history.append(self.prompt)
            elif 'target_objects' in msg:
                self.target_objects = msg['target_objects']
                self.object_colors = self._generate_object_colors()
                logging.info(f"Updated target objects: {self.target_objects}")
            elif 'detection_threshold' in msg:
                self.detection_threshold = float(msg['detection_threshold'])
                logging.info(f"Updated detection threshold: {self.detection_threshold}")
            elif 'pause_video' in msg:
                self.pause_video = msg['pause_video']
                self.pause_image = None
                logging.info(f"{'pausing' if self.pause_video else 'resuming'} processing of incoming video stream")
            elif 'auto_refresh' in msg:
                self.auto_refresh = msg['auto_refresh']
                logging.info(f"{'enabling' if self.auto_refresh else 'disabling'} auto-refresh of model output with prior query")
            elif 'auto_refresh_db' in msg:
                self.auto_refresh_db = msg['auto_refresh_db']
                logging.info(f"{'enabling' if self.auto_refresh_db else 'disabling'} auto-refresh of vector database search results")
            elif 'save_db' in msg:
                if self.db:
                    self.db.db.save()
            elif 'tag_image' in msg:
                self.tag_image = msg['tag_image']
            elif 'max_new_tokens' in msg:
                self.llm(max_new_tokens=int(msg['max_new_tokens']))
            elif 'rag_threshold' in msg:
                self.rag_threshold = float(msg['rag_threshold']) / 100.0
                logging.debug(f"set RAG threshold to {self.rag_threshold}")
                
    def poll_keyboard(self):
        while True:
            try:
                key = input().strip()
                
                if key == 'd' or key == 'l':
                    self.keyboard_prompt = (self.keyboard_prompt + 1) % len(self.prompt_history)
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                elif key == 'a' or key == 'j':
                    self.keyboard_prompt = self.keyboard_prompt - 1
                    if self.keyboard_prompt < 0:
                        self.keyboard_prompt = len(self.prompt_history) - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
                num = int(key)
                
                if num > 0 and num <= len(self.prompt_history):
                    self.keyboard_prompt = num - 1
                    self.prompt = self.prompt_history[self.keyboard_prompt]
                    
            except Exception as err:
                continue
     
    def print_stats(self):
        curr_time = time.perf_counter()
            
        if not hasattr(self, 'start_time'):
            self.start_time = curr_time
        else:
            frame_time = curr_time - self.start_time
            self.start_time = curr_time
            refresh_str = f"{1.0 / frame_time:.2f} FPS ({frame_time*1000:.1f} ms)"
            self.server.send_message({'refresh_rate': refresh_str})
            logging.info(f"refresh rate:  {refresh_str}")

    def start(self):
        """
        Start the webserver & websocket listening in other threads.
        """
        super().start()
        self.server.start()
        return self
        
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['video_input', 'video_output', 'web', 'nanodb'])
    parser.add_argument("--target-objects", nargs='+', default=['car', 'bicycle', 'dog', 'cat'], 
                       help="List of objects to detect and highlight")
    parser.add_argument("--detection-threshold", type=float, default=0.5, 
                       help="Confidence threshold for object detection (0.0 to 1.0)")
    args = parser.parse_args()
    agent = ObjectDetection(**vars(args)).run()
