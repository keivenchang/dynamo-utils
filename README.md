# Dynamo Utils

A collection of utility scripts and configuration files for developing and deploying the NVIDIA Dynamo distributed inference framework.

## ⚠️ DISCLAIMER

**IMPORTANT**: This is an experimental utilities repository and is NOT officially tied to or supported by the ai-dynamo project. These tools are provided as-is without any warranty or official support. Use at your own risk.

This collection of utilities is maintained independently for development convenience and is not part of the official Dynamo project.

## Overview

This repository contains essential development tools, build scripts, and configuration management utilities for working with the Dynamo project. These tools streamline the development workflow, manage container environments, and facilitate testing of the Dynamo inference system.

## Prerequisites

- Docker with GPU support
- Python >= 3.10
- Rust toolchain (for building Dynamo components)
- Git
- jq (for JSON processing in scripts)

## Directory Structure

```
dynamo-utils/
├── compile.sh              # Build and install Dynamo Python packages
├── curl.sh                 # Test models via chat completions API
├── inference.sh            # Launch Dynamo inference services
├── sync_devcontainer.sh   # Sync dev configs across projects
├── devcontainer.json       # VS Code Dev Container configuration
└── devcontainer/           # Dev container specific files
```

## Key Scripts

### Build & Development

#### `compile.sh`
Builds and installs Python packages for the Dynamo distributed inference framework.

```bash
# Development mode (fast build, editable install)
./compile.sh --dev

# Release mode (optimized build)
./compile.sh --release

# Clean Python packages
./compile.sh --python-clean

# Clean Rust build artifacts
./compile.sh --cargo-clean
```

**Packages built:**
- `ai-dynamo-runtime`: Core Rust extensions + Python bindings
- `ai-dynamo`: Complete framework with all components

### Testing & Inference

#### `curl.sh`
Convenient script to test models via the chat completions API with retry logic and response validation.

```bash
# Basic test
./curl.sh --port 8080 --prompt "Hello!"

# Streaming with retry
./curl.sh --stream --retry --prompt "Tell me a story"

# Loop testing with metrics
./curl.sh --loop --metrics --random
```

**Options:**
- `--port`: API server port (default: 8080)
- `--prompt`: User prompt
- `--stream`: Enable streaming responses
- `--retry`: Retry until success
- `--loop`: Run in infinite loop
- `--metrics`: Show performance metrics

#### `inference.sh`
Launches Dynamo inference services (frontend and backend).

```bash
# Run with default framework
./inference.sh

# Run with specific backend
./inference.sh --framework vllm

# Dry run to see commands
./inference.sh --dry-run
```

**Environment variables:**
- `DYN_FRONTEND_PORT`: Frontend port (default: 8080)
- `DYN_BACKEND_PORT`: Backend port (default: 8081)

### Configuration Management

#### `sync_devcontainer.sh`
Automatically syncs development configuration files across multiple Dynamo project directories.

```bash
# Sync configurations
./sync_devcontainer.sh

# Dry run to preview changes
./sync_devcontainer.sh --dryrun

# Force sync regardless of changes
./sync_devcontainer.sh --force

# Silent mode for cron jobs
./sync_devcontainer.sh --silent
```

**How it works - Example:**

Before running `sync_devcontainer.sh`:
```
~/nvidia/
├── dynamo-utils/           # This repository with master config files
│   ├── devcontainer.json   # Master devcontainer config
│   └── sync_devcontainer.sh
├── dynamo1/                # Clone of dynamo repo for feature A
│   └── (existing dynamo files...)
├── dynamo2/                # Clone of dynamo repo for feature B
│   └── (existing dynamo files...)
└── dynamo3/                # Clone of dynamo repo for experiments
    └── (existing dynamo files...)
```

After running `./sync_devcontainer.sh`:
```
~/nvidia/
├── dynamo-utils/           # This repository (unchanged)
│   ├── devcontainer.json
│   └── sync_devcontainer.sh
├── dynamo1/
│   ├── .devcontainer/
│   │   └── [username]/
│   │       └── devcontainer.json  # ← Customized: container name = dynamo1-[username]-devcontainer
│   └── (existing dynamo files...)
├── dynamo2/
│   ├── .devcontainer/
│   │   └── [username]/
│   │       └── devcontainer.json  # ← Customized: container name = dynamo2-[username]-devcontainer
│   └── (existing dynamo files...)
└── dynamo3/
    ├── .devcontainer/
    │   └── [username]/
    │       └── devcontainer.json  # ← Customized: container name = dynamo3-[username]-devcontainer
    └── (existing dynamo files...)
```

Key points:
- Each dynamo directory gets its own copy of configuration files
- The devcontainer.json is customized per directory with unique container names
- This allows working on multiple features/branches simultaneously with isolated containers
- All directories stay in sync with the master configurations in dynamo-utils

**Files synced:**
- `.devcontainer/` configurations

## Environment Setup

The typical workflow for setting up a development environment:

1. Clone the Dynamo repository
2. Use `./compile.sh --dev` to build in development mode
3. Test with `./inference.sh` and `./curl.sh`

## Tips & Best Practices

1. **Development Mode**: Use `./compile.sh --dev` for faster iteration during development
2. **Testing**: Always test API endpoints with `./curl.sh` after starting services
3. **Configuration Sync**: Run `sync_devcontainer.sh` periodically or via cron to keep configs updated
4. **Container Development**: Use the Dev Container for a consistent development environment
5. **Port Conflicts**: Check port availability before running inference services

## Troubleshooting

### Port Already in Use
If you encounter port conflicts when running `inference.sh`:
```bash
# Check what's using the port
lsof -i :8080
# Kill the process or use different ports
DYN_FRONTEND_PORT=8090 DYN_BACKEND_PORT=8091 ./inference.sh
```

### Build Failures
For build issues with `compile.sh`:
```bash
# Clean and rebuild
./compile.sh --cargo-clean
./compile.sh --python-clean
./compile.sh --dev
```

### Container Issues
If Docker container fails to start:
```bash
# Check Docker daemon
docker ps
# Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Contributing

When contributing to this repository:
1. Test scripts thoroughly before committing
2. Update this README if adding new scripts or features
3. Use meaningful commit messages with `--signoff`
