#!/bin/bash

set -euo pipefail

# GPU Reset and Memory Clear Utility
# Usage: ./gpu_reset.sh [--soft|--hard|--driver-reset]

RESET_TYPE="soft"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Reset and clear GPU memory and processes.

OPTIONS:
    -h, --help          Show this help message and exit
    --soft              Soft reset: Kill ML processes only (default)
    --hard              Hard reset: Reset GPU state and kill all CUDA processes
    --driver-reset      Driver reset: Unload and reload NVIDIA drivers (requires sudo)
    --status            Show current GPU status only

DESCRIPTION:
    This script provides different levels of GPU reset:
    - Soft: Kills Python/ML processes to free GPU memory
    - Hard: Resets GPU clocks and kills all CUDA processes  
    - Driver: Unloads and reloads NVIDIA drivers (most aggressive)
EOF
}

show_gpu_status() {
    echo "=== Current GPU Status ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo "❌ nvidia-smi not found"
        exit 1
    fi
    echo
}

soft_reset() {
    echo "🔄 Performing soft GPU reset..."
    
    # Kill common ML framework processes
    echo "Killing Python/ML processes..."
    pkill -f "vllm" 2>/dev/null || true
    pkill -f "torch" 2>/dev/null || true
    pkill -f "tensorflow" 2>/dev/null || true
    pkill -f "pytorch" 2>/dev/null || true
    pkill -f "python3.*--endpoint" 2>/dev/null || true
    
    # Kill multiprocess workers (from your inference script pattern)
    (ps -ef --forest 2>/dev/null | grep multiprocess | awk '{print $2}' | xargs kill 2>/dev/null) || true
    (ps -ef 2>/dev/null | grep "python3.*\/tmp" | awk '{print $2}' | xargs kill 2>/dev/null) || true
    
    echo "✅ Soft reset complete"
}

hard_reset() {
    echo "🔄 Performing hard GPU reset..."
    
    # First do soft reset
    soft_reset
    
    # Kill all processes using NVIDIA devices
    echo "Killing all CUDA processes..."
    if command -v fuser &> /dev/null; then
        sudo fuser -k /dev/nvidia* 2>/dev/null || true
    fi
    
    # Reset GPU clocks and state
    echo "Resetting GPU clocks and state..."
    sudo nvidia-smi -rgc 2>/dev/null || echo "⚠️  Could not reset GPU clocks (may not be supported)"
    
    echo "✅ Hard reset complete"
}

driver_reset() {
    echo "🔄 Performing driver-level GPU reset..."
    echo "⚠️  This will disconnect all applications using the GPU!"
    
    # Unload NVIDIA drivers
    echo "Unloading NVIDIA drivers..."
    sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || echo "⚠️  Some modules may not be loaded"
    
    # Wait a moment
    sleep 2
    
    # Reload NVIDIA drivers
    echo "Reloading NVIDIA drivers..."
    sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
    
    # Restart nvidia-persistenced if it exists
    if systemctl is-active --quiet nvidia-persistenced; then
        sudo systemctl restart nvidia-persistenced
    fi
    
    echo "✅ Driver reset complete"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --soft)
            RESET_TYPE="soft"
            shift
            ;;
        --hard)
            RESET_TYPE="hard"
            shift
            ;;
        --driver-reset)
            RESET_TYPE="driver"
            shift
            ;;
        --status)
            show_gpu_status
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show initial status
show_gpu_status

# Perform reset based on type
case $RESET_TYPE in
    soft)
        soft_reset
        ;;
    hard)
        hard_reset
        ;;
    driver)
        driver_reset
        ;;
esac

# Show final status
echo
echo "=== GPU Status After Reset ==="
show_gpu_status

echo "🎉 GPU reset complete!"
