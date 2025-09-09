#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Load test script for LLM backend with support for streaming and non-streaming requests.
Usage: ./soak_llm.py --duration_sec 60 --workers 50 --requests_per_worker 100
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp


class LoadTestConfig:
    """Configuration for the load test."""

    def __init__(self):
        self.duration_sec = 60
        self.workers = 50
        self.requests_per_worker = 100
        self.port = 8000
        self.model = "Qwen/Qwen3-0.6B"
        self.max_tokens = 20
        self.streaming = False
        self.output_responses = False
        self.log_file = "/tmp/soak_llm.log"
        self.version = "v1_chat_completion"  # Default to chat completions


class LoadTestStats:
    """Statistics tracking for the load test."""

    def __init__(self, config: 'LoadTestConfig' = None):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.end_time = None
        self.results = []
        self.config = config

    def add_result(self, worker_id: int, request_id: int, success: bool,
                   duration: float, status_code: int, response_content: str = ""):
        """Add a request result."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        status = "SUCCESS" if success else f"ERROR({status_code})"

        result = {
            'timestamp': timestamp,
            'worker_id': worker_id,
            'request_id': request_id,
            'duration': duration,
            'status': status,
            'response_content': response_content
        }
        self.results.append(result)

        # Print to stdout in real-time if --output is specified
        if self.config and self.config.output_responses and response_content:
            content = response_content.strip().replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')
            # Only trim for multiple workers to avoid cluttered output
            if self.config.workers > 1 and len(content) > 70:
                original_cleaned = response_content.strip().replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')
                content = content[:70] + f"... ({len(original_cleaned) - 70} more chars)"
            print(f"[{worker_id}-{request_id}] RESPONSE: {content}")

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_requests += 1

    def finalize(self):
        """Finalize statistics."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get actual test duration."""
        end_time = self.end_time or time.time()
        return end_time - self.start_time

    def get_rps(self) -> float:
        """Get requests per second."""
        duration = self.get_duration()
        return self.total_requests / duration if duration > 0 else 0


class StreamingParser:
    """Parser for Server-Sent Events (SSE) streaming responses."""

    @staticmethod
    def parse_streaming_response(response_text: str) -> str:
        """Parse streaming response and extract content."""
        content_parts = []

        for line in response_text.strip().split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    choices = data.get('choices', [])
                    if choices:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content') or delta.get('reasoning_content')
                        if content:
                            content_parts.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        return ''.join(content_parts)


class LoadTestWorker:
    """Worker that performs load test requests."""

    def __init__(self, worker_id: int, config: LoadTestConfig, stats: LoadTestStats):
        self.worker_id = worker_id
        self.config = config
        self.stats = stats
        self.cancelled = False

    async def run(self, session: aiohttp.ClientSession) -> None:
        """Run the worker's requests."""
        for request_id in range(1, self.config.requests_per_worker + 1):
            if self.cancelled:
                break

            try:
                if self.config.streaming:
                    success, duration, status_code, content = await self._make_streaming_request(session, request_id)
                else:
                    success, duration, status_code, content = await self._make_non_streaming_request(session, request_id)

                self.stats.add_result(
                    self.worker_id, request_id, success, duration, status_code, content
                )

            except asyncio.CancelledError:
                print(f"Worker {self.worker_id} cancelled", file=sys.stderr)
                break
            except Exception as e:
                print(f"Worker {self.worker_id} request {request_id} failed: {e}", file=sys.stderr)
                self.stats.add_result(
                    self.worker_id, request_id, False, 0.0, 0, ""
                )

    def cancel(self):
        """Cancel this worker."""
        self.cancelled = True

    def _build_payload_and_url(self, prompt: str, streaming: bool) -> Tuple[Dict, str]:
        """Build the appropriate payload and URL based on the API version."""
        if self.config.version == "v1_completion":
            # Use v1/completions endpoint with prompt-based payload
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": streaming,
                "max_tokens": self.config.max_tokens
            }
            url = f"http://localhost:{self.config.port}/v1/completions"
        else:
            # Use v1/chat/completions endpoint with messages-based payload (default)
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "developer", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": streaming,
                "max_tokens": self.config.max_tokens
            }
            url = f"http://localhost:{self.config.port}/v1/chat/completions"

        return payload, url

    async def _stream_response_realtime(self, response, request_id: int) -> str:
        """Stream response in real-time and output tokens as they arrive."""
        content_parts = []

        print(f"[{self.worker_id}-{request_id}] Streaming response:")

        async for line in response.content:
            line_str = line.decode('utf-8').strip()
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                    choices = data.get('choices', [])
                    if choices:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content') or delta.get('reasoning_content')
                        if content:
                            # Filter out carriage returns and newlines from streaming output
                            filtered_content = content.replace('\n', '').replace('\r', '').replace('\\n', '').replace('\\r', '')
                            content_parts.append(content)  # Keep original for final response
                            # Output filtered content for clean streaming display
                            print(filtered_content, end='', flush=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        print()  # New line after streaming is complete
        return ''.join(content_parts)

    async def _make_streaming_request(self, session: aiohttp.ClientSession, request_id: int) -> Tuple[bool, float, int, str]:
        """Make a streaming request."""
        prompt = f"Write a long and complete story about Lincoln ({request_id} / {self.worker_id})"
        payload, url = self._build_payload_and_url(prompt, streaming=True)

        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(url, json=payload, timeout=timeout) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    content = ""
                    if self.config.output_responses and self.config.workers == 1:
                        # Stream tokens in real-time for single worker
                        content = await self._stream_response_realtime(response, request_id)
                    elif self.config.output_responses:
                        # Parse streaming response normally for multiple workers
                        response_text = await response.text()
                        content = StreamingParser.parse_streaming_response(response_text)
                    else:
                        # Still need to consume the response even if not outputting
                        await response.text()

                    return True, duration, response.status, content
                else:
                    return False, duration, response.status, ""

        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            duration = time.time() - start_time
            print(f"Streaming request error: {e}", file=sys.stderr)
            return False, duration, 0, ""

    async def _make_non_streaming_request(self, session: aiohttp.ClientSession, request_id: int) -> Tuple[bool, float, int, str]:
        """Make a non-streaming request."""
        prompt = f"Write a long and complete story about Lincoln ({request_id} / {self.worker_id})"
        payload, url = self._build_payload_and_url(prompt, streaming=False)

        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(url, json=payload, timeout=timeout) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    content = ""
                    if self.config.output_responses:
                        try:
                            data = await response.json()
                            choices = data.get('choices', [])
                            if choices:
                                if self.config.version == "v1_completion":
                                    # v1/completions format - uses 'content'
                                    content = choices[0].get('text', '')
                                else:
                                    # v1_chat_completion format - uses 'reasoning_content'
                                    message = choices[0].get('message', {})
                                    content = message.get('reasoning_content', '')
                        except (json.JSONDecodeError, KeyError, IndexError):
                            content = ""

                    return True, duration, response.status, content
                else:
                    return False, duration, response.status, ""

        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            duration = time.time() - start_time
            print(f"Non-streaming request error: {e}", file=sys.stderr)
            return False, duration, 0, ""


class LoadTester:
    """Main load tester class."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.stats = LoadTestStats(config)
        self.running = True
        self.workers = []
        self.tasks = []
        self.interrupted = False

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        if not self.interrupted:
            print("\nInterrupt received. Cleaning up workers...")
            self.interrupted = True
            self.running = False
            # Cancel all running tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()

    async def run(self) -> None:
        """Run the load test."""
        print("Starting load test...")
        print(f"Backend URL: localhost:{self.config.port}")
        print(f"Duration: {self.config.duration_sec} seconds")
        print(f"Workers: {self.config.workers}")
        print(f"Requests per worker: {self.config.requests_per_worker}")
        print(f"Total requests: {self.config.workers * self.config.requests_per_worker}")
        print(f"Streaming: {self.config.streaming}")
        print()
        print(f"Results will be stored in: {self.config.log_file}")

        # Create workers
        workers = []
        for worker_id in range(1, self.config.workers + 1):
            worker = LoadTestWorker(worker_id, self.config, self.stats)
            workers.append(worker)

        print(f"Starting {len(workers)} workers...")

        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=self.config.workers * 2)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            try:
                # Create tasks for all workers
                self.tasks = [asyncio.create_task(worker.run(session)) for worker in workers]

                # Monitor progress
                start_time = time.time()
                last_report = 0

                while self.running and time.time() - start_time < self.config.duration_sec:
                    current_time = time.time() - start_time

                    # Check for interrupt first
                    if self.interrupted:
                        break

                    # Report progress every 5 seconds
                    if int(current_time) >= last_report + 5:
                        active_workers = sum(1 for task in self.tasks if not task.done())
                        print(f"Elapsed: {int(current_time)}s, Active workers: {active_workers}")
                        last_report = int(current_time)

                    # Check if all workers completed
                    if all(task.done() for task in self.tasks):
                        print("All workers completed before time limit.")
                        break

                    await asyncio.sleep(0.1)

                if self.interrupted:
                    print("Interrupted by user.")
                elif time.time() - start_time >= self.config.duration_sec:
                    print("Time limit reached. Stopping workers...")

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received during execution.")
                self.running = False
                self.interrupted = True

            # Clean shutdown of remaining workers
            remaining_tasks = [task for task in self.tasks if not task.done()]
            if remaining_tasks:
                print("Cleaning up remaining workers...")
                # Cancel remaining tasks
                for task in remaining_tasks:
                    task.cancel()

                # Wait briefly for cancellation to complete
                try:
                    await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        self._finalize_and_report()

    def _finalize_and_report(self) -> None:
        """Finalize statistics and generate report."""
        self.stats.finalize()

        print("Collecting results...")
        print()

        # Write detailed logs
        self._write_logs()

        # Print summary
        self._print_summary()

    def _write_logs(self) -> None:
        """Write detailed logs to file."""
        try:
            with open(self.config.log_file, 'w') as f:
                for result in self.stats.results:
                    log_entry = (
                        f"{result['timestamp']} "
                        f"worker_{result['worker_id']} "
                        f"request_{result['request_id']} "
                        f"duration_{result['duration']:.3f}s "
                        f"status_{result['status']}"
                    )

                    if self.config.output_responses and result['response_content']:
                        # Truncate long responses for display (only for multiple workers)
                        content = result['response_content'].strip().replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')
                        if self.config.workers > 1 and len(content) > 70:
                            original_cleaned = result['response_content'].strip().replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')
                            content = content[:70] + f"... ({len(original_cleaned) - 70} more chars)"
                        log_entry += f" response={content}"

                    f.write(log_entry + '\n')
        except Exception as e:
            print(f"Error writing logs: {e}", file=sys.stderr)

    def _print_summary(self) -> None:
        """Print test summary."""
        duration = self.stats.get_duration()
        rps = self.stats.get_rps()
        success_rate = (self.stats.successful_requests / self.stats.total_requests * 100) if self.stats.total_requests > 0 else 0

        print("=== LOAD TEST RESULTS ===")
        print(f"Actual duration: {duration:.0f} seconds")
        print(f"Total requests: {self.stats.total_requests}")
        print(f"Successful requests: {self.stats.successful_requests}")
        print(f"Failed requests: {self.stats.failed_requests}")
        print(f"Requests per second: {rps:.0f}")
        if self.stats.total_requests > 0:
            print(f"Success rate: {success_rate:.1f}%")
        print()
        print(f"Detailed logs are in: {self.config.log_file}")
        print("Results saved permanently. No cleanup needed.")


def parse_args() -> LoadTestConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load test the backend with concurrent HTTP requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use defaults (5000 total requests)
  %(prog)s --duration_sec 30 --workers 20    # 30s with 20 workers
  %(prog)s --workers 100 --requests_per_worker 50  # High concurrency
  %(prog)s --port 8080 --duration_sec 120    # Use port 8080 for 2 minutes
  %(prog)s --max-tokens 2000 --workers 10    # Generate longer responses
  %(prog)s --stream --workers 5 --requests_per_worker 2  # Test streaming responses
  %(prog)s --output --workers 1 --requests_per_worker 3  # Show server responses
  %(prog)s --version v1_completion --workers 10  # Use v1/completions endpoint
        """
    )

    parser.add_argument('--duration_sec', '--duration', type=int, default=60,
                        help='Duration in seconds (default: 60)')
    parser.add_argument('--workers', type=int, default=50,
                        help='Number of concurrent workers (default: 50)')
    parser.add_argument('--requests_per_worker', type=int, default=100,
                        help='Requests per worker (default: 100)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Backend port (default: 8000)')
    parser.add_argument('--model', default="Qwen/Qwen3-0.6B",
                        help='Model name (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--max-tokens', type=int, default=20,
                        help='Maximum tokens to generate (default: 20)')
    parser.add_argument('--version', choices=['v1_chat_completion', 'v1_completion'],
                        default='v1_chat_completion',
                        help='API version to use (default: v1_chat_completion)')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming responses')
    parser.add_argument('--output', action='store_true',
                        help='Show server responses in output')

    args = parser.parse_args()

    config = LoadTestConfig()
    config.duration_sec = args.duration_sec
    config.workers = args.workers
    config.requests_per_worker = args.requests_per_worker
    config.port = args.port
    config.model = args.model
    config.max_tokens = args.max_tokens
    config.streaming = args.stream
    config.output_responses = args.output
    config.version = args.version

    return config


async def main():
    """Main entry point."""
    try:
        config = parse_args()
        tester = LoadTester(config)
        await tester.run()

        # Check if we were interrupted
        if tester.interrupted:
            print("Load test interrupted and cleaned up.")
            sys.exit(130)  # Standard exit code for SIGINT

    except KeyboardInterrupt:
        print("\nLoad test interrupted and cleaned up.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
