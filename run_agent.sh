#!/bin/bash

# Multi-Container Agent Orchestrator Runner
# Manages the lifecycle of the agentic multimodal system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
COMPOSE_FILE="podman-compose.yaml"
CONFIG_FILE="agent_config.json"
LOG_DIR="./logs"
DATA_DIR="./data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."

    if ! command -v podman &> /dev/null; then
        error "podman is not installed"
        exit 1
    fi

    if ! command -v podman-compose &> /dev/null; then
        warning "podman-compose not found, trying docker-compose"
        if ! command -v docker-compose &> /dev/null; then
            error "Neither podman-compose nor docker-compose found"
            exit 1
        fi
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="podman-compose"
    fi

    success "Dependencies checked"
}

# Initialize directories and files
init_environment() {
    log "Initializing environment..."

    # Create directories
    mkdir -p "$LOG_DIR" "$DATA_DIR" "$DATA_DIR/knowledge_base" "$DATA_DIR/diagrams"

    # Check if config exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi

    # Create sample knowledge base if it doesn't exist
    if [[ ! -d "$DATA_DIR/knowledge_base/documents" ]]; then
        mkdir -p "$DATA_DIR/knowledge_base/documents"
        cat > "$DATA_DIR/knowledge_base/documents/sample.txt" << EOF
# Sample Technical Documentation

## 12-Pin Connector Specifications
Standard automotive connector with power and signal pins.
- Pin 1-4: Power (12V)
- Pin 5-8: Ground
- Pin 9-12: Signal lines
- Torque specification: 5-7 Nm

## ABS Warning Light Troubleshooting
1. Check brake fluid level
2. Inspect wheel speed sensors
3. Scan for diagnostic codes
4. Check ABS fuse
5. Inspect wiring harness for damage

## Connector Replacement Procedure
1. Disconnect battery
2. Remove old connector
3. Clean contact area
4. Apply dielectric grease
5. Install new connector
6. Torque to specification
7. Test connection
EOF
        log "Created sample knowledge base"
    fi

    success "Environment initialized"
}

# Build containers
build_containers() {
    log "Building containers..."

    $COMPOSE_CMD -f "$COMPOSE_FILE" build

    success "Containers built"
}

# Start the agent system
start_agent() {
    log "Starting multi-container agent system..."

    # Start core services first
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d agent-orchestrator redis

    # Wait for orchestrator to be ready
    log "Waiting for orchestrator to be ready..."
    timeout=60
    while ! curl -sf http://localhost:8080/health &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            error "Orchestrator failed to start within timeout"
            return 1
        fi
    done

    success "Agent system started successfully"
    log "Orchestrator API available at: http://localhost:8080"
    log "Check logs with: podman logs agent-orchestrator"
}

# Stop the agent system
stop_agent() {
    log "Stopping agent system..."

    # Stop all containers
    $COMPOSE_CMD -f "$COMPOSE_FILE" down

    # Stop any running VLM or RAG containers
    podman stop live-llava rag-llm 2>/dev/null || true

    success "Agent system stopped"
}

# Show system status
status() {
    log "Agent System Status:"
    echo

    # Check orchestrator
    if curl -sf http://localhost:8080/health &> /dev/null; then
        success "✓ Orchestrator: Running"
    else
        error "✗ Orchestrator: Not running"
    fi

    # Check containers
    echo
    log "Container Status:"
    podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(agent-|live-|rag-)" || echo "No agent containers running"

    # Check resources
    echo
    log "Resource Usage:"
    podman stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | head -10 || echo "No containers running"
}

# View logs
logs() {
    local service=${1:-agent-orchestrator}
    log "Showing logs for $service..."

    case $service in
        "orchestrator"|"agent")
            podman logs -f agent-orchestrator
            ;;
        "vlm"|"llava")
            podman logs -f live-llava 2>/dev/null || echo "VLM container not running"
            ;;
        "rag"|"llm")
            podman logs -f rag-llm 2>/dev/null || echo "RAG container not running"
            ;;
        "redis")
            podman logs -f agent-redis
            ;;
        *)
            podman logs -f "$service"
            ;;
    esac
}

# Run single processing cycle
single_cycle() {
    log "Running single processing cycle..."

    if ! curl -sf http://localhost:8080/health &> /dev/null; then
        error "Orchestrator not running. Start the agent first."
        exit 1
    fi

    # Trigger single cycle via API
    curl -X POST http://localhost:8080/cycle 2>/dev/null || {
        error "Failed to trigger processing cycle"
        exit 1
    }

    success "Processing cycle triggered"
}

# Manual container management
start_vlm() {
    log "Starting VLM container manually..."

    podman run -d --rm --name live-llava \
        --device nvidia.com/gpu=all \
        --group-add keep-groups \
        --network host \
        --shm-size=8g \
        --security-opt label=disable \
        -e HF_HOME=/data/hf-cache \
        -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
        -e PYTHONPATH=/opt/clip_trt:/opt/NanoLLM:/opt/NanoDB:/opt/faiss_lite \
        -v "$DATA_DIR:/data" \
        -v /run/user/1000/pulse:/run/user/1000/pulse \
        --device /dev/bus/usb \
        --device /dev/video0 \
        -p 8554:8554 \
        docker.io/dustynv/nano_llm:r36.4.0 \
        python3 vlm_service.py \
            --model Efficient-Large-Model/VILA1.5-3b \
            --video-device /dev/video0 \
            --host 0.0.0.0 \
            --port 8554 \
            --auto-start

    success "VLM container started"
}

start_rag() {
    log "Starting RAG container manually..."

    podman run -d --rm --name rag-llm \
        --device nvidia.com/gpu=all \
        --group-add keep-groups \
        --network host \
        --shm-size=8g \
        --security-opt label=disable \
        -e HF_HOME=/data/hf-cache \
        -e PYTHONPATH=/opt/NanoLLM:/opt/NanoDB:/opt/faiss_lite \
        -v "$DATA_DIR:/data" \
        -p 8555:8555 \
        docker.io/dustynv/nano_llm:r36.4.0 \
        python3 rag_llm_service.py \
            --model microsoft/DialoGPT-medium \
            --kb-path /data/knowledge_base \
            --host 0.0.0.0 \
            --port 8555

    success "RAG container started"
}

# Show help
show_help() {
    cat << EOF
Multi-Container Agent Orchestrator

Usage: $0 [COMMAND]

Commands:
    start       Start the agent system
    stop        Stop the agent system
    restart     Restart the agent system
    status      Show system status
    build       Build containers
    logs [svc]  Show logs (orchestrator|vlm|rag|redis)
    cycle       Run single processing cycle
    start-vlm   Start VLM container manually
    start-rag   Start RAG container manually
    init        Initialize environment
    help        Show this help

Examples:
    $0 start                 # Start the agent system
    $0 logs orchestrator     # View orchestrator logs
    $0 cycle                 # Trigger single processing cycle
    $0 status                # Check system status

EOF
}

# Main execution
main() {
    case "${1:-help}" in
        "start")
            check_dependencies
            init_environment
            start_agent
            ;;
        "stop")
            stop_agent
            ;;
        "restart")
            stop_agent
            sleep 2
            check_dependencies
            init_environment
            start_agent
            ;;
        "status")
            status
            ;;
        "build")
            check_dependencies
            build_containers
            ;;
        "logs")
            logs "$2"
            ;;
        "cycle")
            single_cycle
            ;;
        "start-vlm")
            start_vlm
            ;;
        "start-rag")
            start_rag
            ;;
        "init")
            init_environment
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Execute main function
main "$@"