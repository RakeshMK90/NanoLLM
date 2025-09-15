#!/usr/bin/env python3
"""
Multi-Container Agent Orchestrator
Coordinates VLM observations with RAG+LLM processing
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import aiohttp
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class Observation:
    """Structured observation from VLM"""
    timestamp: float
    content: str
    confidence: float
    detected_objects: List[str]
    suggested_actions: List[str]

@dataclass
class AgentDecision:
    """Decision made by the agent brain"""
    should_query_rag: bool
    should_fetch_diagrams: bool
    rag_query: str
    diagram_query: str
    priority: int  # 1=high, 2=medium, 3=low

class AgentOrchestrator:
    """Main orchestrator that manages container lifecycle and decision making"""

    def __init__(self, config_file: str = "agent_config.json"):
        self.config = self.load_config(config_file)
        self.container_states = {}
        self.observation_history = []
        self.session = None

    def load_config(self, config_file: str) -> Dict:
        """Load agent configuration"""
        default_config = {
            "vlm_container": {
                "name": "live-llava",
                "image": "docker.io/dustynv/nano_llm:r36.4.0",
                "api_port": 8554
            },
            "rag_container": {
                "name": "rag-llm",
                "image": "docker.io/dustynv/nano_llm:r36.4.0",
                "api_port": 8555
            },
            "orchestrator": {
                "decision_threshold": 0.7,
                "max_observations": 100,
                "container_timeout": 30
            }
        }

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return default_config

    async def start_session(self):
        """Initialize HTTP session for container communication"""
        self.session = aiohttp.ClientSession()

    async def stop_session(self):
        """Clean up HTTP session"""
        if self.session:
            await self.session.close()

    def get_container_status(self, container_name: str) -> ContainerState:
        """Check if a podman container is running"""
        try:
            result = subprocess.run(
                ["podman", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            if container_name in result.stdout:
                return ContainerState.RUNNING
            else:
                return ContainerState.STOPPED
        except subprocess.CalledProcessError:
            return ContainerState.ERROR

    async def start_vlm_container(self) -> bool:
        """Start the VLM container for live video analysis"""
        container_name = self.config["vlm_container"]["name"]

        if self.get_container_status(container_name) == ContainerState.RUNNING:
            logger.info(f"VLM container {container_name} already running")
            return True

        cmd = [
            "podman", "run", "-d", "--rm", "--name", container_name,
            "--device", "nvidia.com/gpu=all",
            "--group-add", "keep-groups",
            "--network", "host",
            "--shm-size=8g",
            "--security-opt", "label=disable",
            "-e", "HF_HOME=/data/hf-cache",
            "-e", "PULSE_SERVER=unix:/run/user/1000/pulse/native",
            "-e", "PYTHONPATH=/opt/clip_trt:/opt/NanoLLM:/opt/NanoDB:/opt/faiss_lite",
            "-v", "/home/rakesh/jetson-containers/data:/data",
            "-v", "/run/user/1000/pulse:/run/user/1000/pulse",
            "--device", "/dev/bus/usb",
            "--device", "/dev/video0",
            "-p", f"{self.config['vlm_container']['api_port']}:8554",
            self.config["vlm_container"]["image"],
            "python3", "-m", "nano_llm.agents.video_query",
            "--api=mlc",
            "--model", "Efficient-Large-Model/VILA1.5-3b",
            "--max-context-len", "256",
            "--max-new-tokens", "32",
            "--video-input", "/dev/video0",
            "--video-output", f"webrtc://@:{self.config['vlm_container']['api_port']}/output",
            "--web-host", "0.0.0.0",
            "--web-port", "8554"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Started VLM container: {container_name}")

            # Wait for container to be ready
            await asyncio.sleep(5)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start VLM container: {e.stderr}")
            return False

    async def stop_container(self, container_name: str) -> bool:
        """Stop a podman container"""
        try:
            subprocess.run(["podman", "stop", container_name], check=True)
            logger.info(f"Stopped container: {container_name}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Failed to stop container: {container_name}")
            return False

    async def get_vlm_observation(self) -> Optional[Observation]:
        """Get structured observation from VLM container"""
        vlm_port = self.config["vlm_container"]["api_port"]

        try:
            # This would be the actual API endpoint from your VLM container
            async with self.session.get(f"http://localhost:{vlm_port}/observation") as response:
                if response.status == 200:
                    data = await response.json()
                    return Observation(
                        timestamp=time.time(),
                        content=data.get("content", ""),
                        confidence=data.get("confidence", 0.0),
                        detected_objects=data.get("detected_objects", []),
                        suggested_actions=data.get("suggested_actions", [])
                    )
        except Exception as e:
            logger.error(f"Failed to get VLM observation: {e}")

        return None

    def make_decision(self, observation: Observation) -> AgentDecision:
        """Agent brain: decide what actions to take based on observation"""
        decision = AgentDecision(
            should_query_rag=False,
            should_fetch_diagrams=False,
            rag_query="",
            diagram_query="",
            priority=3
        )

        # Decision logic based on observation content
        content_lower = observation.content.lower()

        # High priority technical issues
        if any(keyword in content_lower for keyword in ["warning", "error", "fault", "problem"]):
            decision.should_query_rag = True
            decision.priority = 1
            decision.rag_query = f"troubleshooting {observation.content}"

        # Connector or hardware identification
        if any(keyword in content_lower for keyword in ["connector", "pin", "cable", "port"]):
            decision.should_fetch_diagrams = True
            decision.should_query_rag = True
            decision.priority = 2
            decision.rag_query = f"connector specifications {observation.content}"
            decision.diagram_query = observation.content

        # Procedure-related observations
        if any(keyword in content_lower for keyword in ["procedure", "step", "install", "replace"]):
            decision.should_query_rag = True
            decision.priority = 2
            decision.rag_query = f"procedure {observation.content}"

        # General technical questions
        if observation.confidence > self.config["orchestrator"]["decision_threshold"]:
            if not decision.should_query_rag:
                decision.should_query_rag = True
                decision.rag_query = observation.content

        logger.info(f"Decision: RAG={decision.should_query_rag}, Diagrams={decision.should_fetch_diagrams}, Priority={decision.priority}")
        return decision

    async def start_rag_container(self) -> bool:
        """Start the RAG+LLM container"""
        container_name = self.config["rag_container"]["name"]

        if self.get_container_status(container_name) == ContainerState.RUNNING:
            logger.info(f"RAG container {container_name} already running")
            return True

        cmd = [
            "podman", "run", "-d", "--rm", "--name", container_name,
            "--device", "nvidia.com/gpu=all",
            "--group-add", "keep-groups",
            "--network", "host",
            "--shm-size=8g",
            "--security-opt", "label=disable",
            "-e", "HF_HOME=/data/hf-cache",
            "-e", "PYTHONPATH=/opt/NanoLLM:/opt/NanoDB:/opt/faiss_lite",
            "-v", "/home/rakesh/jetson-containers/data:/data",
            "-p", f"{self.config['rag_container']['api_port']}:8555",
            self.config["rag_container"]["image"],
            "python3", "-m", "nano_llm.chat",
            "--api=mlc",
            "--model", "microsoft/DialoGPT-medium",  # 4B quantized model
            "--quantization", "q4f16_ft",
            "--max-context-len", "2048",
            "--max-new-tokens", "256",
            "--web-host", "0.0.0.0",
            "--web-port", "8555"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Started RAG container: {container_name}")
            await asyncio.sleep(5)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start RAG container: {e.stderr}")
            return False

    async def query_rag_system(self, query: str, fetch_diagrams: bool = False) -> Dict[str, Any]:
        """Query the RAG system and optionally fetch diagrams"""
        rag_port = self.config["rag_container"]["api_port"]

        # Prepare the enhanced prompt with RAG context
        enhanced_prompt = f"""
        Based on the technical observation: {query}

        Please provide:
        1. Relevant technical information
        2. Step-by-step procedures if applicable
        3. Safety considerations
        4. Required tools or parts
        """

        try:
            # Query the LLM with RAG context
            payload = {
                "prompt": enhanced_prompt,
                "max_tokens": 256,
                "temperature": 0.7
            }

            async with self.session.post(f"http://localhost:{rag_port}/chat", json=payload) as response:
                if response.status == 200:
                    llm_response = await response.json()

                    result = {
                        "llm_response": llm_response,
                        "diagrams": []
                    }

                    # Optionally fetch diagrams from NanoDB
                    if fetch_diagrams:
                        # This would query your NanoDB service
                        result["diagrams"] = await self.fetch_diagrams(query)

                    return result

        except Exception as e:
            logger.error(f"Failed to query RAG system: {e}")

        return {"llm_response": {"error": "RAG query failed"}, "diagrams": []}

    async def fetch_diagrams(self, query: str) -> List[str]:
        """Fetch relevant diagrams from NanoDB"""
        # Placeholder for NanoDB integration
        # This would connect to your diagram database
        logger.info(f"Fetching diagrams for: {query}")
        return []

    async def process_cycle(self):
        """Main processing cycle: VLM -> Decision -> RAG+LLM"""
        try:
            # Start VLM container
            if not await self.start_vlm_container():
                logger.error("Failed to start VLM container")
                return

            # Get observation from VLM
            observation = await self.get_vlm_observation()
            if not observation:
                logger.warning("No observation received from VLM")
                return

            # Store observation
            self.observation_history.append(observation)
            if len(self.observation_history) > self.config["orchestrator"]["max_observations"]:
                self.observation_history.pop(0)

            logger.info(f"Observation: {observation.content} (confidence: {observation.confidence})")

            # Make decision
            decision = self.make_decision(observation)

            # Execute decision
            if decision.should_query_rag or decision.should_fetch_diagrams:
                # Stop VLM to free resources
                await self.stop_container(self.config["vlm_container"]["name"])

                # Start RAG container
                if await self.start_rag_container():
                    result = await self.query_rag_system(
                        decision.rag_query,
                        decision.should_fetch_diagrams
                    )

                    logger.info("RAG Response received")
                    logger.info(f"LLM: {result['llm_response']}")

                    if result["diagrams"]:
                        logger.info(f"Diagrams: {result['diagrams']}")

                    # Stop RAG container
                    await self.stop_container(self.config["rag_container"]["name"])
                else:
                    logger.error("Failed to start RAG container")

        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")

    async def run_continuous(self, cycle_interval: int = 10):
        """Run the agent in continuous mode"""
        logger.info("Starting continuous agent operation")

        await self.start_session()

        try:
            while True:
                await self.process_cycle()
                await asyncio.sleep(cycle_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down agent")
        finally:
            # Clean up containers
            await self.stop_container(self.config["vlm_container"]["name"])
            await self.stop_container(self.config["rag_container"]["name"])
            await self.stop_session()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Container Agent Orchestrator")
    parser.add_argument("--config", default="agent_config.json", help="Configuration file")
    parser.add_argument("--interval", type=int, default=10, help="Processing cycle interval in seconds")
    parser.add_argument("--single-cycle", action="store_true", help="Run single cycle instead of continuous")

    args = parser.parse_args()

    orchestrator = AgentOrchestrator(args.config)

    if args.single_cycle:
        asyncio.run(orchestrator.process_cycle())
    else:
        asyncio.run(orchestrator.run_continuous(args.interval))