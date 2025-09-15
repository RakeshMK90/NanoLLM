# Multi-Container Orchestrated Agent System

A sophisticated agentic multimodal system that orchestrates VLM (Vision-Language Model) observations with RAG (Retrieval-Augmented Generation) + LLM processing using podman containers.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VLM Container │    │ Agent Brain     │    │ RAG+LLM         │
│   (Live LLaVA)  │───▶│ (Orchestrator)  │───▶│ Container       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Data Storage    │
                       │ (FAISS + NanoDB)│
                       └─────────────────┘
```

## How It Works

### 1. Agent Brain (Orchestrator)
The `agent_orchestrator.py` is the decision-making center that:
- **Manages container lifecycle**: Starts/stops VLM and RAG containers as needed
- **Analyzes observations**: Processes structured observations from the VLM
- **Makes intelligent decisions**: Determines when to query RAG database or fetch diagrams
- **Orchestrates workflow**: Coordinates the entire multi-step process

Decision Logic:
- **High priority**: Warning/error keywords trigger immediate RAG queries
- **Connector identification**: Hardware keywords trigger both RAG and diagram retrieval
- **Procedure requests**: Process-related observations trigger procedural document retrieval

### 2. VLM Container (Live LLaVA)
The `vlm_service.py` provides:
- **Live video analysis**: Processes camera feed with VILA1.5-3b model
- **Structured observations**: Extracts technical elements, objects, and suggested actions
- **REST API**: Provides `/observation` endpoint for real-time data
- **Confidence scoring**: Rates observation quality and relevance

Key Features:
- Processes every 30th frame (1-second intervals at 30fps) to avoid overwhelming
- Extracts technical keywords (connectors, warnings, procedures)
- Provides structured output with detected objects and suggested actions

### 3. RAG + LLM Container
The `rag_llm_service.py` combines:
- **FAISS vector search**: Retrieves relevant technical documentation
- **Quantized LLM**: 4B parameter model for response generation
- **Knowledge base**: Technical procedures, safety information, specifications
- **NanoDB integration**: Fetches relevant diagrams and visual aids

## Container Resource Management

**Key Constraint**: Only one GPU-intensive container runs at a time to optimize resource usage.

**Workflow**:
1. **VLM Phase**: Live LLaVA analyzes video → generates observation
2. **Decision Phase**: Orchestrator decides on actions
3. **RAG Phase**: VLM stops → RAG+LLM starts → processes query → stops
4. **Repeat**: Cycle continues with next observation

## Quick Start

### Prerequisites
```bash
# Install podman and podman-compose
sudo apt install podman podman-compose

# Ensure GPU access is configured
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:11.0-base nvidia-smi
```

### Setup and Run
```bash
# Make script executable
chmod +x run_agent.sh

# Initialize environment and start system
./run_agent.sh start

# Check status
./run_agent.sh status

# View logs
./run_agent.sh logs orchestrator
```

### Manual Container Control
```bash
# Start VLM container directly
./run_agent.sh start-vlm

# Start RAG container directly
./run_agent.sh start-rag

# Run single processing cycle
./run_agent.sh cycle
```

## Configuration

### Agent Configuration (`agent_config.json`)
```json
{
  "vlm_container": {
    "model": "Efficient-Large-Model/VILA1.5-3b",
    "max_context_len": 256,
    "max_new_tokens": 32
  },
  "rag_container": {
    "model": "microsoft/DialoGPT-medium",
    "quantization": "q4f16_ft",
    "max_context_len": 2048
  },
  "orchestrator": {
    "decision_threshold": 0.7,
    "cycle_interval": 10,
    "max_concurrent_containers": 1
  }
}
```

### Decision Rules
The orchestrator uses keyword-based decision making:
- **High Priority**: `["warning", "error", "fault", "failure", "alarm", "critical"]`
- **Connectors**: `["connector", "pin", "cable", "port", "socket", "plug"]`
- **Procedures**: `["procedure", "step", "install", "replace", "repair"]`
- **Safety**: `["safety", "hazard", "danger", "caution", "risk"]`

## API Endpoints

### Orchestrator API (Port 8080)
- `GET /health` - System health check
- `POST /cycle` - Trigger single processing cycle
- `GET /status` - Get system status
- `GET /history` - View processing history

### VLM API (Port 8554)
- `GET /observation` - Get latest structured observation
- `GET /observations/history` - Get observation history
- `POST /start` - Start video processing
- `POST /stop` - Stop video processing

### RAG API (Port 8555)
- `POST /query` - Query RAG system with retrieval
- `POST /chat` - Direct LLM chat
- `GET /history` - View query history
- `GET /kb/stats` - Knowledge base statistics

## Example Workflow

1. **Video Input**: Camera shows "ABS warning icon detected"
2. **VLM Processing**: Live LLaVA generates structured observation:
   ```json
   {
     "content": "ABS warning icon detected on dashboard",
     "confidence": 0.9,
     "detected_objects": ["warning", "indicator"],
     "suggested_actions": ["investigate_issue", "check_documentation"]
   }
   ```
3. **Agent Decision**: Orchestrator decides:
   - High priority (warning detected)
   - Should query RAG: Yes
   - Should fetch diagrams: Yes
   - RAG query: "troubleshooting ABS warning icon detected on dashboard"

4. **Container Orchestration**:
   - Stop VLM container
   - Start RAG+LLM container
   - Query knowledge base for ABS troubleshooting procedures
   - Generate response with step-by-step instructions
   - Stop RAG container

5. **Response**:
   ```json
   {
     "response": "ABS warning indicates brake system issue. Steps: 1. Check brake fluid level 2. Inspect wheel sensors 3. Scan diagnostic codes...",
     "confidence": 0.9,
     "retrieved_docs": [{"id": "abs_troubleshooting", "content": "..."}]
   }
   ```

## Monitoring and Debugging

### View System Status
```bash
./run_agent.sh status
```

### Monitor Logs
```bash
# Orchestrator logs
./run_agent.sh logs orchestrator

# VLM logs (when running)
./run_agent.sh logs vlm

# RAG logs (when running)
./run_agent.sh logs rag
```

### Health Checks
- Orchestrator: `curl http://localhost:8080/health`
- VLM (when active): `curl http://localhost:8554/health`
- RAG (when active): `curl http://localhost:8555/health`

## Extending the System

### Adding New Decision Rules
Edit `agent_config.json` decision rules section or modify the `make_decision()` method in `agent_orchestrator.py`.

### Adding Knowledge Base Content
Place documents in `./data/knowledge_base/documents/` directory. The system will automatically index them.

### Adding Diagram Support
Implement NanoDB integration in `rag_llm_service.py` and place diagrams in `./data/diagrams/`.

### Custom Models
Update container configurations in `agent_config.json` to use different VLM or LLM models.

## Troubleshooting

### Common Issues
1. **GPU Access**: Ensure `nvidia-container-toolkit` is installed
2. **Container Conflicts**: Only one GPU container should run at a time
3. **Port Conflicts**: Check that ports 8080, 8554, 8555 are available
4. **Video Device**: Ensure `/dev/video0` exists and is accessible

### Debug Commands
```bash
# Check podman GPU access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:11.0-base nvidia-smi

# Check container status
podman ps -a

# View all logs
podman logs agent-orchestrator
```

## Performance Optimization

- **Frame Processing**: VLM processes every 30th frame to balance latency and accuracy
- **Resource Management**: Single GPU container constraint prevents memory conflicts
- **Caching**: Redis caching for frequent queries and observations
- **Model Quantization**: 4-bit quantized models for optimal performance/memory balance

This system provides a robust foundation for multimodal agentic applications with efficient resource management and intelligent decision-making capabilities.