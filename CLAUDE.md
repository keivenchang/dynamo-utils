# NVIDIA Dynamo Projects - Instructions

> **Note**: For coding conventions, style guidelines, and development practices, refer to `.cursorrules` in this directory.

---

# PART 1: OPERATIONAL PROCEDURES

## Meta Instructions

### Remember This
When the user says "remember this", document it in this CLAUDE.md file.

### Commit Policy
**NEVER auto-commit changes without explicit user approval.** Always wait for the user to explicitly request a commit before running git commit commands.

### Permission Policy
- No need to ask permission when running any read-only operations, such as `echo`, `cat`, `tail`, `head`, `grep`, `egrep`, `ls`, `uname`, `soak_fe.py` or `curl` commands
- When executing `docker exec ... bash -c "<command> ..." and the <command> is one of the read-only operations, just execute the command, no need to ask for permission.

## Environment Setup

### All Projects Overview

The `~/nvidia/` directory contains multiple projects:

- **dynamo1, dynamo2, dynamo3, dynamo4**: Multiple working branches of the Dynamo repository
- **dynamo_ci**: Main CI/testing repository for Dynamo
- **dynamo-utils**: Build automation scripts and utilities (this directory)

### Docker Container Naming

When running `docker ps`, VS Code/Cursor dev container images follow this naming pattern:

- `vsc-dynamo1-*` → `~/nvidia/dynamo1`
- `vsc-dynamo2-*` → `~/nvidia/dynamo2`
- `vsc-dynamo3-*` → `~/nvidia/dynamo3`
- `vsc-dynamo4-*` → `~/nvidia/dynamo4`
- `vsc-dynamo_ci-*` → `~/nvidia/dynamo_ci`

The `vsc-` prefix indicates VS Code/Cursor dev containers, and the part after it matches the directory name.

**Note**: Container names (like `epic_satoshi`, `distracted_shockley`) are transient and should not be documented.

### Host-Container Directory Mapping

The `dynamo-utils` directory on the host is mapped to the container's `_` directory:

- **Host**: `~/nvidia/dynamo-utils/`
- **Container**: `~/dynamo/_/`

Example: A file at `~/nvidia/dynamo-utils/notes/metrics-vllm.log` on the host appears at `~/dynamo/_/notes/metrics-vllm.log` inside the container.

### Backup File Convention

When creating backup files, use the naming format: `<filename>.<YYYY-MM-DD>.bak`

Example:
```bash
cp dynamo_docker_builder.py dynamo_docker_builder.py.2025-10-18.bak
```

This convention:
- Makes backup dates immediately visible
- Allows multiple backups from different dates
- Is automatically ignored by .gitignore (*.bak pattern)

## Running Inference Servers

### Collecting Prometheus Metrics from Inference Server

This procedure collects Prometheus metrics from a running inference server inside a Docker container. Repeat this process for different frameworks (VLLM, SGLANG, TRTLLM) to compare metrics.

**Prerequisites**:
- Docker container running
- Container must have dynamo project mounted

**Steps**:

1. **Start inference server in background**:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-<framework>.log 2>&1 &"
```

2. **Monitor inference server startup** (wait ~30 seconds):
```bash
docker exec <container_name> bash -c "tail -20 /tmp/inference-<framework>.log"
```
Look for: "added model model_name=..." indicating the model is ready.

3. **Run soak test** to generate some load:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"
```
Wait for: "All requests completed successfully."

4. **Collect metrics and save to _ directory**:
```bash
docker exec <container_name> bash -c "curl -s localhost:8081/metrics > ~/dynamo/_/notes/metrics-<framework>.log"
```

**Example for VLLM**:
```bash
# Find container name
docker ps --format "table {{.Names}}\t{{.Image}}" | grep dynamo1

# Start inference (example container: epic_satoshi)
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-vllm.log 2>&1 &"

# Wait and check log
sleep 30 && docker exec <container_name> bash -c "tail -20 /tmp/inference-vllm.log"

# Run soak test
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"

# Collect metrics
docker exec <container_name> bash -c "mkdir -p ~/dynamo/_/notes && curl -s localhost:8081/metrics > ~/dynamo/_/notes/metrics-vllm.log"
```

**Output**:
- Metrics saved to: `~/nvidia/dynamo-utils/notes/metrics-<framework>.log` (on host)
- Typical size: ~200-600 lines

**Repeat for other frameworks**:
- SGLANG: Save to `metrics-sglang.log`
- TRTLLM: Save to `metrics-trtllm.log`

**Cleanup after collection**:
```bash
# Kill inference processes
docker exec <container_name> bash -c "ps aux | grep -E '(inference|sglang|vllm|trtllm|dynamo)' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"

# Verify ports freed
docker exec <container_name> bash -c "ss -tlnp | grep -E '(8000|8081)' || echo 'Ports freed'"
```

### Prometheus Metrics Comparison: VLLM vs SGLANG vs TRTLLM

**Collection Date**: 2025-10-21 (Updated with correct test parameters)

**Test Parameters**: `--max-tokens 1000 --requests_per_worker 5`

#### Summary

Successfully collected and compared Prometheus metrics from three inference frameworks running on the same model (Qwen/Qwen3-0.6B) with 5 requests × 1000 tokens each:

| Framework | Total Lines | Unique Metrics | Dynamo Metrics | Request Duration | TTFT | Throughput |
|-----------|-------------|----------------|----------------|------------------|------|------------|
| **TRTLLM** | 220 | 5 (trtllm:*) | 53 lines | **2.19s** | **10.3ms** | **456 tok/s** |
| **SGLANG** | 224 | 25 (sglang:*) | 35 lines | 2.77s | N/A | 361 tok/s |
| VLLM | 527 | 25 (vllm:*) | 53 lines | 5.94s | 20.6ms | 168 tok/s |

**Key Result**: TRTLLM is 2.7x faster than VLLM and 1.3x faster than SGLANG for this workload.

#### Key Findings

**1. Metrics Structure Differences**

**VLLM** provides detailed metrics:
- **25 unique metrics** (`vllm:*` - excluding `_created` metadata)
- Detailed token-level histograms (prompt, generation, iteration)
- Multiple latency breakdowns (TTFT, inter-token, per-output-token, queue time, prefill time, decode time)
- Cache configuration: KV cache usage %, prefix cache hits/queries
- Comprehensive observability for debugging

**SGLANG** provides moderate metrics:
- **25 unique metrics** (`sglang:*`)
- Focus on queue management (6 different queue types tracked)
- Per-stage request latency, KV transfer speed/latency
- Cache hit rate, token usage, utilization, speculative decoding metrics
- Good balance of observability

**TRTLLM** provides minimal focused metrics:
- **5 unique metrics** (`trtllm:*` - excluding `_created` metadata)
- Core performance only: E2E latency, TTFT, time-per-output-token, queue time, success counter
- Extremely lightweight instrumentation - 5x fewer metrics than VLLM/SGLANG
- Lowest overhead among all frameworks

**Dynamo common metrics** (shared across all):
- `dynamo_component_*` metrics (uptime, NATS client, KV stats, request handling)
- 53 lines for VLLM/TRTLLM, 35 lines for SGLANG

**2. Performance Characteristics**

**TRTLLM** (FASTEST):
- Time to first token (TTFT): 10.3ms avg (2.0x faster than VLLM)
- Time per output token: 2.26ms avg (2.6x faster than VLLM)
- Request processing: 2.19s avg for 1000 tokens (2.7x faster than VLLM)
- Throughput: 456 tokens/sec (5000 tokens / 10.97s)

**SGLANG** (MIDDLE):
- Request processing: 2.77s avg for 1000 tokens (2.1x faster than VLLM)
- Reported generation throughput: 460 tokens/sec
- Actual throughput: 361 tokens/sec (5000 tokens / 13.84s)
- Note: Reported metric (460) vs actual (361) shows 27% discrepancy

**VLLM** (SLOWEST):
- Time to first token (TTFT): 20.6ms avg
- Time per output token: ~5.94ms avg
- Request processing: 5.94s avg for 1000 tokens
- Throughput: 168 tokens/sec (5000 tokens / 29.70s)

**3. Token Processing**

All frameworks processed:
- **Prompt tokens**: 22 per request (110 total across 5 requests)
- **Generation tokens**: 1000 per request (5000 total across 5 requests)
- **Total tokens**: 1022 per request (5110 total)

**VLLM**:
- KV cache blocks available: 4,834
- GPU memory utilization: 20%
- Request finished due to: `length` limit (max_tokens=1000)

**SGLANG**:
- KV cache: Different architecture from VLLM/TRTLLM

**4. Key Observations**

- **Performance ranking**: TRTLLM (456 tok/s) > SGLANG (361 tok/s) > VLLM (168 tok/s)
- **Speed ratios**: TRTLLM is 2.7x faster than VLLM, 1.3x faster than SGLANG
- **TTFT comparison**: TRTLLM 10.3ms < VLLM 20.6ms (SGLANG: no TTFT data)
- **Metrics count**: VLLM and SGLANG both have 25 metrics, TRTLLM has only 5 (5x fewer)
- **Observability vs Performance**: TRTLLM achieves 2.7x better performance with 5x fewer metrics
- **SGLANG throughput**: Reported metric (460 tok/s) is 27% higher than actual (361 tok/s)

**5. Common Dynamo Metrics**

All frameworks showed consistent Dynamo component behavior:
- NATS client connected (state=1)
- Backend uptime: TRTLLM 119s, VLLM 278s, SGLANG 79s
- Request bytes: 1,253 bytes (identical across all frameworks)
- Response bytes: ~18,200 bytes (TRTLLM 18,200, VLLM 18,129, SGLANG 18,219)
- Active endpoints: 2 (VLLM, TRTLLM), 1 (SGLANG)

#### Recommendations

**Use TRTLLM when**:
- **Maximum performance is critical** (2.7x faster than VLLM)
- Ultra-low latency needed (10.3ms TTFT)
- High throughput required (456 tokens/sec)
- Minimal observability overhead acceptable
- Production workloads prioritizing speed

**Use SGLANG when**:
- Good balance of speed and features (2.1x faster than VLLM)
- Moderate throughput needs (361 tokens/sec)
- Alternative to TRTLLM when feature set is needed

**Use VLLM when**:
- Detailed observability and debugging required
- Most comprehensive metrics suite (527 lines)
- Development and troubleshooting scenarios
- Acceptable performance for lower-throughput workloads

#### Performance Summary

**Winner: TRTLLM**
- 2.7x faster than VLLM (2.19s vs 5.94s per 1000 tokens)
- 2.0x faster TTFT (10.3ms vs 20.6ms)
- 2.7x higher throughput (456 vs 168 tokens/sec)
- Best choice for production performance

**Runner-up: SGLANG**
- 2.1x faster than VLLM (2.77s vs 5.94s per 1000 tokens)
- Good middle ground between speed and features
- 361 tokens/sec actual throughput

**Third: VLLM**
- Excellent observability with most detailed metrics
- Best for development and debugging
- 168 tokens/sec throughput

## Testing Commands

> **Note**: General pytest guidelines (including `--basetemp=/tmp/pytest_temp`) are in `.cursorrules`.

**Quick Test (Single Framework)**:
```bash
python3 dynamo_docker_builder.py --sanity-check-only --framework sglang --force-run --email <email>
```

**Parallel Build with Skip**:
```bash
python3 dynamo_docker_builder.py --skip-build-if-image-exists --parallel --force-run --email <email>
```

**Full Build**:
```bash
python3 dynamo_docker_builder.py --parallel --force-run --email <email>
```

## Important Reminders

- Never commit sensitive information (credentials, tokens, etc.)
- Always test changes locally before pushing
- Use meaningful commit messages
- Review diffs before committing
- Consider backward compatibility when making changes

---

## Additional Documentation

For detailed information about Python utilities and scripts in this directory, see:
- **README.md**: Comprehensive documentation for all Python scripts and tools
- **.cursorrules**: Coding conventions, style guidelines, and development practices
