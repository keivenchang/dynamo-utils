"""
Dynamo utilities package.

Shared constants and utilities for dynamo Docker management scripts.
"""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker  # type: ignore[import-untyped]
except ImportError:
    docker = None  # type: ignore[assignment]

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# GitPython is required - hard error if not installed
try:
    import git
except ImportError as e:
    raise ImportError("GitPython is required. Install with: pip install gitpython") from e

# Supported frameworks
# Used by V2 for default framework list and validation
FRAMEWORKS_UPPER = ["VLLM", "SGLANG", "TRTLLM"]

# DEPRECATED: V1 and retag script reference only - kept for backward compatibility
# V2 uses FRAMEWORKS_UPPER directly
FRAMEWORKS = [f.lower() for f in FRAMEWORKS_UPPER]
FRAMEWORK_NAMES = {"vllm": "VLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}

def normalize_framework(framework: str) -> str:
    """Normalize framework name to canonical lowercase form. DEPRECATED: V1/retag only."""
    return framework.lower()

def get_framework_display_name(framework: str) -> str:
    """Get display name for framework. DEPRECATED: V1/retag only."""
    normalized = normalize_framework(framework)
    return FRAMEWORK_NAMES.get(normalized, normalized.upper())


# Used by retag script only. V1 and V2 do not use these dataclasses.
@dataclass
class DynamoImageInfo:
    """Dynamo-specific Docker image information.

    DEPRECATION: retag script only. V1 and V2 do not use this.
    """
    version: Optional[str] = None      # Parsed version (e.g., "0.1.0.dev.ea07d51fc")
    framework: Optional[str] = None    # Framework name (vllm, sglang, trtllm)
    target: Optional[str] = None       # Target type (local-dev, dev, etc.)
    latest_tag: Optional[str] = None   # Corresponding latest tag

    def matches_sha(self, sha: str) -> bool:
        """Check if this image matches the specified SHA."""
        return bool(self.version and sha in self.version)

    def is_framework_image(self) -> bool:
        """Check if this has framework information."""
        return self.framework is not None

    def get_latest_tag(self, repository: str = "dynamo") -> str:
        """Get the latest tag for this dynamo image."""
        if self.latest_tag:
            return self.latest_tag
        if self.framework:
            if self.target == "dev":
                # dev target maps to just latest-framework (no -dev suffix)
                return f"{repository}:latest-{self.framework}"
            elif self.target:
                # other targets like local-dev keep the suffix
                return f"{repository}:latest-{self.framework}-{self.target}"
            else:
                return f"{repository}:latest-{self.framework}"
        return f"{repository}:latest"


@dataclass
class DockerImageInfo:
    """Comprehensive Docker image information.

    DEPRECATION: retag script only. V1 and V2 do not use this.
    """
    name: str                    # Full image name (repo:tag)
    repository: str              # Repository name
    tag: str                     # Tag name
    image_id: str               # Docker image ID
    created_at: str             # Creation timestamp
    size_bytes: int             # Size in bytes
    size_human: str             # Human readable size
    labels: Dict[str, str]      # Image labels

    # Dynamo-specific information (optional)
    dynamo_info: Optional[DynamoImageInfo] = None

    def matches_sha(self, sha: str) -> bool:
        """Check if this image matches the specified SHA."""
        return bool(self.dynamo_info and self.dynamo_info.matches_sha(sha))

    def is_dynamo_image(self) -> bool:
        """Check if this is a dynamo image."""
        return self.repository in ["dynamo", "dynamo-base"]

    def is_dynamo_framework_image(self) -> bool:
        """Check if this is a dynamo framework image."""
        return bool(self.is_dynamo_image() and self.dynamo_info and self.dynamo_info.is_framework_image())

    def get_latest_tag(self) -> str:
        """Get the latest tag for this image."""
        if self.dynamo_info:
            return self.dynamo_info.get_latest_tag(self.repository)
        return f"{self.repository}:latest"


def get_terminal_width(padding: int = 2, default: int = 118) -> int:
    """
    Detect terminal width using multiple methods in order of preference.

    This function tries several approaches to detect the terminal width,
    which is useful in various environments including PTY/TTY contexts:

    1. Check $COLUMNS environment variable (set by interactive shells)
    2. Try 'tput cols' command (works in more environments than ioctl)
    3. Use shutil.get_terminal_size() (ioctl-based, may return default 80)
    4. Fall back to provided default

    Args:
        padding: Number of characters to subtract from detected width (default: 2)
        default: Default width to use if detection fails (default: 118, i.e., 120 - 2)

    Returns:
        Terminal width in columns, minus the specified padding

    Examples:
        >>> # Get terminal width with default 2-char padding
        >>> width = get_terminal_width()
        >>>
        >>> # Get terminal width with custom padding
        >>> width = get_terminal_width(padding=4)
        >>>
        >>> # Get terminal width with custom default fallback
        >>> width = get_terminal_width(default=78)  # 80 - 2
    """
    term_width = None

    try:
        # Method 1: Check $COLUMNS environment variable (set by shell)
        columns_env = os.environ.get('COLUMNS')
        if columns_env and columns_env.isdigit():
            term_width = int(columns_env) - padding

        # Method 2: Try tput cols (works in more environments than ioctl)
        if term_width is None:
            try:
                result = subprocess.run(
                    ['tput', 'cols'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    term_width = int(result.stdout.strip()) - padding
            except Exception:
                pass

        # Method 3: Use shutil.get_terminal_size() (ioctl-based)
        if term_width is None:
            term_width = shutil.get_terminal_size().columns - padding

    except Exception:
        term_width = default

    # Final fallback to default
    if term_width is None:
        term_width = default

    return term_width


class BaseUtils:
    """Base class for all utility classes with common logger and cmd functionality"""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Set up logger with class name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Create custom formatter that handles DRYRUN prefix and shows class/method
        class DryRunFormatter(logging.Formatter):
            def __init__(self, dry_run_instance) -> None:
                super().__init__()
                self.dry_run_instance = dry_run_instance
            
            def format(self, record: logging.LogRecord) -> str:
                if self.dry_run_instance.verbose:
                    # Verbose mode: show location info
                    location = f"{record.name}.{record.funcName}" if record.funcName != '<module>' else record.name
                    prefix = "DRYRUN" if self.dry_run_instance.dry_run else ""
                    if prefix:
                        return f"{prefix} {record.levelname} - [{location}] {record.getMessage()}"
                    else:
                        return f"{record.levelname} - [{location}] {record.getMessage()}"
                else:
                    # Simple mode: just the message
                    return record.getMessage()
        
        formatter = DryRunFormatter(self)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def cmd(self, command: List[str], return_tuple: bool = False, **kwargs: Any) -> Any:
        """Execute command with dry-run support.
        
        Args:
            command: Command to execute as list of strings
            return_tuple: If True, return (success, stdout, stderr). If False, return CompletedProcess
            **kwargs: Additional arguments passed to subprocess.run()
            
        Returns:
            CompletedProcess object (default) or (success, stdout, stderr) tuple if return_tuple=True
        """
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        self.logger.debug(f"+ {cmd_str}")
        
        if self.dry_run:
            if return_tuple:
                self.logger.info(f"DRY RUN: Would execute: {' '.join(command)}")
                return True, "", ""
            else:
                # Return a mock completed process in dry-run mode
                mock_result: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(command, 0)
                mock_result.stdout = ""
                mock_result.stderr = ""
                return mock_result
        
        # Set default kwargs for tuple interface
        if return_tuple:
            kwargs.setdefault('capture_output', True)
            kwargs.setdefault('text', True)
            kwargs.setdefault('check', False)
        
        try:
            result = subprocess.run(command, **kwargs)
            
            if return_tuple:
                success = result.returncode == 0
                return success, result.stdout or "", result.stderr or ""
            else:
                return result
                
        except Exception as e:
            if return_tuple:
                return False, "", str(e)
            else:
                # Re-raise for CompletedProcess interface
                raise
    


class DynamoRepositoryUtils(BaseUtils):
    """Utilities for Dynamo repository operations including composite SHA calculation."""

    def __init__(self, repo_path: Any, dry_run: bool = False, verbose: bool = False):
        """
        Initialize DynamoRepositoryUtils.

        Args:
            repo_path: Path to Dynamo repository (Path object or str)
            dry_run: Dry-run mode
            verbose: Verbose logging
        """
        super().__init__(dry_run, verbose)
        from pathlib import Path
        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path

    def generate_composite_sha(self, full_hash: bool = False) -> str:
        """
        Generate composite SHA from container directory files.

        This creates a SHA256 hash of all relevant files in the container directory,
        excluding documentation, temporary files, etc. This hash can be used to
        determine if a rebuild is needed.

        Args:
            full_hash: If True, return full 64-char SHA. If False, return first 12 chars.

        Returns:
            SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": container directory doesn't exist
            - "NO_FILES": no relevant files found
            - "ERROR": error during calculation
        """
        import hashlib
        import tempfile
        from pathlib import Path

        container_dir = self.repo_path / "container"
        if not container_dir.exists():
            self.logger.warning(f"Container directory not found: {container_dir}")
            return "NO_CONTAINER_DIR"

        # Excluded patterns (matching V1)
        excluded_extensions = {'.md', '.rst', '.log', '.bak', '.tmp', '.swp', '.swo', '.orig', '.rej'}
        excluded_filenames = {'README', 'CHANGELOG', 'LICENSE', 'NOTICE', 'AUTHORS', 'CONTRIBUTORS'}
        excluded_specific = {'launch_message.txt'}

        # Collect files to hash
        files_to_hash = []
        for file_path in sorted(container_dir.rglob('*')):
            if not file_path.is_file():
                continue
            # Skip hidden files
            if any(part.startswith('.') for part in file_path.relative_to(container_dir).parts):
                continue
            # Skip excluded extensions
            if file_path.suffix.lower() in excluded_extensions:
                continue
            # Skip excluded names
            if file_path.stem.upper() in excluded_filenames:
                continue
            # Skip specific files
            if file_path.name.lower() in excluded_specific:
                continue
            files_to_hash.append(file_path.relative_to(self.repo_path))

        if not files_to_hash:
            self.logger.warning("No files found to hash in container directory")
            return "NO_FILES"

        self.logger.debug(f"Hashing {len(files_to_hash)} files from container directory")

        # Calculate composite SHA
        try:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                try:
                    for file_rel_path in files_to_hash:
                        full_path = self.repo_path / file_rel_path
                        if full_path.exists():
                            temp_file.write(str(file_rel_path).encode('utf-8'))
                            temp_file.write(b'\n')
                            with open(full_path, 'rb') as f:
                                temp_file.write(f.read())
                            temp_file.write(b'\n')

                    temp_file.flush()
                    with open(temp_path, 'rb') as f:
                        sha_full = hashlib.sha256(f.read()).hexdigest()
                        result = sha_full if full_hash else sha_full[:12]
                        self.logger.debug(f"Composite Docker SHA: {result}")
                        return result
                finally:
                    temp_path.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Error calculating composite SHA: {e}")
            return "ERROR"

    def get_stored_composite_sha(self) -> str:
        """
        Get stored composite SHA from file.

        Returns:
            Stored SHA string, or empty string if not found
        """
        sha_file = self.repo_path / ".last_build_composite_sha"
        if sha_file.exists():
            stored = sha_file.read_text().strip()
            self.logger.debug(f"Found stored composite SHA: {stored[:12]}")
            return stored
        self.logger.debug("No stored composite SHA found")
        return ""

    def store_composite_sha(self, sha: str) -> None:
        """
        Store current composite SHA to file.

        Args:
            sha: Composite Docker SHA to store
        """
        sha_file = self.repo_path / ".last_build_composite_sha"
        sha_file.write_text(sha)
        self.logger.info(f"Stored composite SHA: {sha[:12]}")

    def check_if_rebuild_needed(self, force_run: bool = False) -> bool:
        """
        Check if rebuild is needed based on composite SHA comparison.

        Compares current composite SHA with stored SHA to determine if
        container files have changed since last build.

        Args:
            force_run: If True, proceed with rebuild even if SHA unchanged

        Returns:
            True if rebuild is needed, False otherwise
        """
        self.logger.info("\nChecking if rebuild is needed based on file changes...")
        self.logger.info(f"Composite Docker SHA file: {self.repo_path}/.last_build_composite_sha")

        # Generate current composite SHA (full hash, not truncated)
        current_sha = self.generate_composite_sha(full_hash=True)
        if current_sha in ("NO_CONTAINER_DIR", "NO_FILES", "ERROR"):
            self.logger.warning(f"Failed to generate composite SHA: {current_sha}")
            return True  # Assume rebuild needed

        # Get stored composite SHA
        stored_sha = self.get_stored_composite_sha()

        if stored_sha:
            if current_sha == stored_sha:
                if force_run:
                    self.logger.info(f"Composite Docker SHA unchanged ({current_sha[:12]}) but --force-run specified - proceeding")
                    return True
                else:
                    self.logger.info(f"Composite Docker SHA unchanged ({current_sha[:12]}) - skipping rebuild")
                    self.logger.info("Use --force-run to force rebuild")
                    return False  # No rebuild needed
            else:
                self.logger.info("Composite Docker SHA changed:")
                self.logger.info(f"  Previous: {stored_sha[:12]}")
                self.logger.info(f"  Current:  {current_sha[:12]}")
                self.logger.info("Rebuild needed")
                self.store_composite_sha(current_sha)
                return True
        else:
            self.logger.info("No previous composite SHA found - rebuild needed")
            self.store_composite_sha(current_sha)
            return True


class DockerUtils(BaseUtils):
    """Unified Docker utility class with comprehensive image management."""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        super().__init__(dry_run, verbose)

        if docker is None:
            self.logger.error("Docker package not found. Install with: pip install docker")
            raise ImportError("docker package required")

        # Initialize Docker client
        try:
            self.logger.debug("Equivalent: docker version")
            self.client = docker.from_env()
            self.client.ping()
            self.logger.debug("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_float = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float = size_float / 1024.0
        return f"{size_float:.1f} PB"
    
    def _parse_dynamo_image(self, image_name: str) -> Optional[DynamoImageInfo]:
        """Parse dynamo image name to extract framework and version info."""
        # Pattern for dynamo images: (dynamo|dynamo-base):v{version}-{framework}-{target}
        # Examples: 
        #   dynamo:v0.1.0.dev.ea07d51fc-sglang-local-dev
        #   dynamo-base:v0.1.0.dev.ea07d51fc-vllm-dev
        pattern = r'^(?:dynamo|dynamo-base):v(.+?)-([^-]+)(?:-(.+))?$'
        match = re.match(pattern, image_name)
        
        if not match:
            return None
        
        version_part, framework, target = match.groups()

        # Validate framework
        normalized_framework = normalize_framework(framework)
        if normalized_framework not in FRAMEWORKS:
            return None

        return DynamoImageInfo(
            version=version_part,
            framework=normalized_framework,
            target=target or "",
            latest_tag=None  # Will be computed later if needed
        )
    
    def get_image_info(self, image_name: str) -> Optional[DockerImageInfo]:
        """Get comprehensive information about a Docker image.

        DEPRECATION: V1 + retag script only. V2 uses docker.from_env() directly.
        """
        self.logger.debug(f"Equivalent: docker inspect {image_name}")
        
        try:
            image = self.client.images.get(image_name)
            
            # Parse repository and tag
            if ':' in image_name:
                repository, tag = image_name.split(':', 1)
            else:
                repository = image_name
                tag = 'latest'
            
            # Get basic image info
            size_bytes = image.attrs.get('Size', 0)
            created_at = image.attrs.get('Created', '')
            labels = image.attrs.get('Config', {}).get('Labels') or {}
            
            # Parse dynamo-specific info
            dynamo_info = self._parse_dynamo_image(image_name)
            
            return DockerImageInfo(
                name=image_name,
                repository=repository,
                tag=tag,
                image_id=image.id,  # Full ID
                created_at=created_at,
                size_bytes=size_bytes,
                size_human=self._format_size(size_bytes),
                labels=labels,
                dynamo_info=dynamo_info
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get image info for {image_name}: {e}")
            return None
    
    def list_images(self, name_filter: Optional[str] = None) -> List[DockerImageInfo]:
        """List all Docker images with optional name filtering.

        DEPRECATION: retag script only. V1 and V2 do not use this.

        Returns:
            List of DockerImageInfo objects sorted by creation date (newest first)
        """
        self.logger.debug("Equivalent: docker images --format table")
        
        try:
            images = []
            for image in self.client.images.list():
                for tag in image.tags:
                    if tag and (not name_filter or name_filter in tag):
                        image_info = self.get_image_info(tag)
                        if image_info:
                            images.append(image_info)
            
            # Sort by creation date (newest first)
            images.sort(key=lambda x: x.created_at, reverse=True)
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to list images: {e}")
            return []
    
    def list_dynamo_images(self, framework: Optional[str] = None, target: Optional[str] = None, sha: Optional[str] = None) -> List[DockerImageInfo]:
        """List dynamo framework images with optional filtering.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        # Search for both dynamo and dynamo-base images
        dynamo_images = []
        for prefix in ["dynamo:", "dynamo-base:"]:
            images = self.list_images(name_filter=prefix)
            dynamo_images.extend([img for img in images if img.is_dynamo_image()])
        
        # Apply filters
        if framework:
            framework = normalize_framework(framework)
            dynamo_images = [img for img in dynamo_images
                           if img.dynamo_info and img.dynamo_info.framework == framework]
        
        if target:
            dynamo_images = [img for img in dynamo_images 
                           if img.dynamo_info and img.dynamo_info.target == target]
        
        if sha:
            dynamo_images = [img for img in dynamo_images if img.matches_sha(sha)]
        
        return dynamo_images
    
    def tag_image(self, source_tag: str, target_tag: str) -> bool:
        """Tag a Docker image.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        self.logger.debug(f"Equivalent: docker tag {source_tag} {target_tag}")
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would execute: docker tag {source_tag} {target_tag}")
            return True
        
        try:
            source_image = self.client.images.get(source_tag)
            
            # Parse target tag
            if ':' in target_tag:
                repository, tag = target_tag.split(':', 1)
            else:
                repository = target_tag
                tag = 'latest'
            
            source_image.tag(repository, tag)
            self.logger.info(f"✓ Tagged: {source_tag} -> {target_tag}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to tag {source_tag} -> {target_tag}: {e}")
            return False
    
    def retag_to_latest(self, images: List[DockerImageInfo]) -> Dict[str, int]:
        """Retag multiple images to their latest tags.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        results = {'success': 0, 'failed': 0}
        
        for image in images:
            if image.is_dynamo_framework_image():
                latest_tag = image.get_latest_tag()
                if self.tag_image(image.name, latest_tag):
                    results['success'] += 1
                else:
                    results['failed'] += 1
        
        return results
    
    def filter_unused_build_args(self, docker_command: str) -> str:
        """Remove unused --build-arg flags from Docker build commands for base images.

        DEPRECATION: V1 only. V2 does not use this.

        Base images (dynamo-base) don't use most build arguments. Removing unused
        args helps Docker recognize when builds are truly identical.
        """
        import re
        
        if not re.search(r'--tag\s+dynamo-base:', docker_command):
            # Only filter base image builds
            return docker_command
            
        # List of build args that are typically unused by base images
        unused_args = {
            'PYTORCH_VERSION', 'CUDA_VERSION', 'PYTHON_VERSION',
            'FRAMEWORK_VERSION', 'TARGET_ARCH', 'BUILD_TYPE'
        }
        
        # Split command into parts
        parts = docker_command.split()
        filtered_parts = []
        filtered_args = []
        
        i = 0
        while i < len(parts):
            if parts[i] == '--build-arg' and i + 1 < len(parts):
                arg_name = parts[i + 1].split('=')[0]
                if arg_name in unused_args:
                    filtered_args.append(arg_name)
                    i += 2  # Skip both --build-arg and its value
                else:
                    filtered_parts.extend([parts[i], parts[i + 1]])
                    i += 2
            else:
                filtered_parts.append(parts[i])
                i += 1
        
        if filtered_args and self.verbose:
            self.logger.info(f"Filtered {len(filtered_args)} unused base image build args: {', '.join(sorted(filtered_args))}")
        
        return ' '.join(filtered_parts)
    
    def normalize_command(self, docker_command: str) -> str:
        """Normalize Docker command by removing whitespace and sorting build args.

        DEPRECATION: V1 only. V2 does not use this.

        Helps identify functionally identical commands with different formatting.
        """
        
        # Remove extra whitespace and normalize
        normalized = ' '.join(docker_command.split())
        
        # Sort build args to make commands with same args but different order equivalent
        # Find all --build-arg KEY=VALUE pairs
        build_args = []
        other_parts = []
        
        parts = normalized.split()
        i = 0
        while i < len(parts):
            if parts[i] == '--build-arg' and i + 1 < len(parts):
                build_args.append(f"--build-arg {parts[i + 1]}")
                i += 2
            else:
                other_parts.append(parts[i])
                i += 1
        
        # Sort build args for consistent ordering
        build_args.sort()
        
        # Reconstruct command with sorted build args
        if build_args:
            # Insert sorted build args after 'docker build' but before other args
            docker_build_idx = -1
            for idx, part in enumerate(other_parts):
                if part == 'build':
                    docker_build_idx = idx
                    break
            
            if docker_build_idx >= 0:
                result_parts = other_parts[:docker_build_idx + 1] + build_args + other_parts[docker_build_idx + 1:]
            else:
                result_parts = other_parts + build_args
        else:
            result_parts = other_parts
        
        return ' '.join(result_parts)
    
    def extract_base_image_from_command(self, docker_cmd: str) -> str:
        """Extract the base/FROM image from docker build command arguments"""
        import re
        
        # Look for --build-arg DYNAMO_BASE_IMAGE=... (framework-specific builds)
        match = re.search(r'--build-arg\s+DYNAMO_BASE_IMAGE=([^\s]+)', docker_cmd)
        if match:
            return match.group(1)
        
        # Look for --build-arg BASE_IMAGE=... and BASE_IMAGE_TAG=... (base builds)
        base_image_match = re.search(r'--build-arg\s+BASE_IMAGE=([^\s]+)', docker_cmd)
        base_tag_match = re.search(r'--build-arg\s+BASE_IMAGE_TAG=([^\s]+)', docker_cmd)
        
        if base_image_match and base_tag_match:
            return f"{base_image_match.group(1)}:{base_tag_match.group(1)}"
        elif base_image_match:
            return base_image_match.group(1)
        
        # Look for --build-arg DEV_BASE=... (local-dev builds)
        dev_base_match = re.search(r'--build-arg\s+DEV_BASE=([^\s]+)', docker_cmd)
        if dev_base_match:
            return dev_base_match.group(1)
        
        # Return empty string if no base image found
        return ""
    
    def extract_image_tag_from_command(self, docker_cmd: str) -> str:
        """
        Extract the output tag from docker build command --tag argument.
        Returns the tag string, or empty string if no tag found.
        Raises error if multiple tags are found (should not happen after get_build_commands validation).
        """
        import re
        
        # Find all --tag arguments in the command
        tags = re.findall(r'--tag\s+([^\s]+)', docker_cmd)
        
        if len(tags) == 0:
            return ""
        elif len(tags) == 1:
            return tags[0]
        else:
            # This should not happen if get_build_commands validation is working
            self.logger.error(f"Multiple --tag arguments found in command: {tags}")
            return tags[0]  # Return first tag as fallback


# Git utilities using GitPython API (NO subprocess calls)
class GitUtils(BaseUtils):
    """Git utilities using GitPython API only - NO subprocess calls to git.

    Provides clean API for git operations without any subprocess calls.
    All operations use GitPython's native API.

    Example:
        git_utils = GitUtils(repo_path="/path/to/repo")
        commits = git_utils.get_recent_commits(max_count=50)
        git_utils.checkout(commit_sha)
    """

    def __init__(self, repo_path: Any, dry_run: bool = False, verbose: bool = False):
        """Initialize GitUtils.

        Args:
            repo_path: Path to git repository (Path object or str)
            dry_run: Dry-run mode
            verbose: Verbose logging
        """
        super().__init__(dry_run, verbose)

        from pathlib import Path
        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path

        try:
            self.repo = git.Repo(self.repo_path)
            self.logger.debug(f"Initialized git repo at {self.repo_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize git repository at {self.repo_path}: {e}")
            raise

    def get_current_branch(self) -> Optional[str]:
        """Get current branch name.

        Returns:
            Branch name or None if detached HEAD
        """
        try:
            if self.repo.head.is_detached:
                return None
            return self.repo.active_branch.name
        except Exception as e:
            self.logger.error(f"Failed to get current branch: {e}")
            return None

    def get_current_commit(self) -> str:
        """Get current commit SHA.

        Returns:
            Full commit SHA (40 characters)
        """
        return self.repo.head.commit.hexsha

    def checkout(self, ref: str) -> bool:
        """Checkout a specific commit, branch, or tag using GitPython API.

        Args:
            ref: Commit SHA, branch name, or tag name

        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would checkout {ref}")
            return True

        try:
            self.repo.git.checkout(ref)
            self.logger.debug(f"Checked out {ref}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to checkout {ref}: {e}")
            return False

    def get_commit(self, sha: str) -> Optional[Any]:
        """Get commit object by SHA using GitPython API.

        Args:
            sha: Commit SHA (full or short)

        Returns:
            GitPython commit object or None if not found
        """
        try:
            return self.repo.commit(sha)
        except Exception as e:
            self.logger.error(f"Failed to get commit {sha}: {e}")
            return None

    def get_recent_commits(self, max_count: int = 50, branch: str = 'main') -> List[Any]:
        """Get recent commits from a branch using GitPython API.

        Args:
            max_count: Maximum number of commits to retrieve
            branch: Branch name (default: 'main')

        Returns:
            List of GitPython commit objects
        """
        try:
            commits = list(self.repo.iter_commits(branch, max_count=max_count))
            self.logger.debug(f"Retrieved {len(commits)} commits from {branch}")
            return commits
        except Exception as e:
            self.logger.error(f"Failed to get commits from {branch}: {e}")
            return []

    def get_commit_info(self, commit: Any) -> Dict[str, Any]:
        """Extract information from a commit object.

        Args:
            commit: GitPython commit object

        Returns:
            Dictionary with commit information
        """
        from datetime import datetime

        return {
            'sha_full': commit.hexsha,
            'sha_short': commit.hexsha[:9],
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'committer_name': commit.committer.name,
            'committer_email': commit.committer.email,
            'date': datetime.fromtimestamp(commit.committed_date),
            'message': commit.message.strip(),
            'message_first_line': commit.message.split('\n')[0] if commit.message else '',
            'parents': [p.hexsha for p in commit.parents]
        }

    def is_dirty(self) -> bool:
        """Check if repository has uncommitted changes.

        Returns:
            True if there are uncommitted changes, False otherwise
        """
        return self.repo.is_dirty()

    def get_untracked_files(self) -> List[str]:
        """Get list of untracked files.

        Returns:
            List of untracked file paths
        """
        return self.repo.untracked_files

    def get_tags(self) -> List[str]:
        """Get all repository tags.

        Returns:
            List of tag names
        """
        return [tag.name for tag in self.repo.tags]

    def get_branches(self, remote: bool = False) -> List[str]:
        """Get list of branches.

        Args:
            remote: If True, return remote branches. If False, return local branches.

        Returns:
            List of branch names
        """
        if remote:
            return [ref.name for ref in self.repo.remote().refs]
        else:
            return [head.name for head in self.repo.heads]


# GitHub API utilities

@dataclass
class FailedCheck:
    """Information about a failed CI check"""
    name: str
    job_url: str
    run_id: str
    duration: str
    is_required: bool = False
    error_summary: Optional[str] = None


@dataclass
class RunningCheck:
    """Information about a running CI check"""
    name: str
    check_url: str
    is_required: bool = False
    elapsed_time: Optional[str] = None


@dataclass
class PRInfo:
    """Pull request information"""
    number: int
    title: str
    url: str
    state: str
    is_merged: bool
    review_decision: Optional[str]
    mergeable_state: str
    unresolved_conversations: int
    ci_status: Optional[str]
    base_ref: Optional[str] = None
    created_at: Optional[str] = None
    has_conflicts: bool = False
    conflict_message: Optional[str] = None
    blocking_message: Optional[str] = None
    failed_checks: List['FailedCheck'] = field(default_factory=list)
    running_checks: List['RunningCheck'] = field(default_factory=list)
    rerun_url: Optional[str] = None


class GitHubAPIClient:
    """GitHub API client with automatic token detection and rate limit handling.

    Features:
    - Automatic token detection (--token arg > GITHUB_TOKEN env > GitHub CLI config)
    - Request/response handling with proper error messages
    - Support for parallel API calls with ThreadPoolExecutor

    Example:
        client = GitHubAPIClient()
        pr_data = client.get("/repos/owner/repo/pulls/123")
    """

    @staticmethod
    def get_github_token_from_cli() -> Optional[str]:
        """Get GitHub token from GitHub CLI configuration.

        Reads the token from ~/.config/gh/hosts.yml if available.

        Returns:
            GitHub token string, or None if not found
        """
        if not HAS_YAML:
            return None

        try:
            gh_config_path = Path.home() / '.config' / 'gh' / 'hosts.yml'
            if gh_config_path.exists():
                with open(gh_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'github.com' in config:
                        github_config = config['github.com']
                        if 'oauth_token' in github_config:
                            return github_config['oauth_token']
                        if 'users' in github_config:
                            for user, user_config in github_config['users'].items():
                                if 'oauth_token' in user_config:
                                    return user_config['oauth_token']
        except Exception:
            pass
        return None

    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub API client.

        Args:
            token: GitHub personal access token. If not provided, will try:
                   1. GITHUB_TOKEN environment variable
                   2. GitHub CLI config (~/.config/gh/hosts.yml)
        """
        if not HAS_REQUESTS:
            raise ImportError("requests package required for GitHub API client. Install with: pip install requests")

        # Token priority: 1) provided token, 2) environment variable, 3) GitHub CLI config
        self.token = token or os.environ.get('GITHUB_TOKEN') or self.get_github_token_from_cli()
        self.base_url = "https://api.github.com"
        self.headers = {'Accept': 'application/vnd.github.v3+json'}

        if self.token:
            self.headers['Authorization'] = f'token {self.token}'

        # Cache for job logs (two-tier: memory + disk)
        self._job_log_cache: Dict[str, Optional[str]] = {}
        self._cache_dir = Path.home() / ".cache" / "dynamo-utils" / "job-logs"

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict]:
        """Make GET request to GitHub API.

        Args:
            endpoint: API endpoint (e.g., "/repos/owner/repo/pulls/123")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response as dict, or None if request failed
        """
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)

            if response.status_code == 403:
                # Check if it's a rate limit error
                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
                    raise Exception("GitHub API rate limit exceeded. Use --token argument or set GITHUB_TOKEN environment variable.")
                else:
                    raise Exception(f"GitHub API returned 403 Forbidden: {response.text}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"GitHub API request failed for {endpoint}: {e}")

    def has_token(self) -> bool:
        """Check if a GitHub token is configured."""
        return self.token is not None

    def get_ci_status(self, owner: str, repo: str, sha: str, pr_number: Optional[int] = None) -> Optional[str]:
        """Get CI status for a commit by checking PR checks.

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            pr_number: Optional PR number for faster lookup

        Returns:
            CI status string ('passed', 'failed', 'running'), or None if unavailable
        """
        # If we don't have PR number, try to find it
        if not pr_number:
            # Try to find PR from commit
            endpoint = f"/repos/{owner}/{repo}/commits/{sha}/pulls"
            try:
                pulls = self.get(endpoint)
                if pulls and len(pulls) > 0:
                    pr_number = pulls[0]['number']
                else:
                    # Fallback to legacy status API
                    endpoint = f"/repos/{owner}/{repo}/commits/{sha}/status"
                    data = self.get(endpoint)
                    if data:
                        state = data.get('state')
                        if state == 'success':
                            return 'passed'
                        elif state == 'failure':
                            return 'failed'
                        elif state == 'pending':
                            return 'running'
                    return None
            except Exception:
                return None

        # Use gh CLI to get check status (most reliable)
        try:
            import subprocess
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # gh pr checks returns non-zero if there are failures or pending checks
            # We still want to parse the output even if return code is non-zero
            if not result.stdout:
                return None

            # Parse output - count pass/fail/pending
            lines = result.stdout.lower().split('\n')
            has_fail = any('fail' in line for line in lines)
            has_pending = any('pending' in line for line in lines)

            if has_fail:
                return 'failed'
            elif has_pending:
                return 'running'
            else:
                return 'passed'

        except Exception:
            return None

    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        """Get full PR details including mergeable status.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            PR details as dict, or None if request failed
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            return self.get(endpoint)
        except Exception:
            return None

    def count_unresolved_conversations(self, owner: str, repo: str, pr_number: int) -> int:
        """Count unresolved conversation threads in a PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Count of unresolved conversations (approximated as top-level comments)
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        try:
            comments = self.get(endpoint)
            if not comments:
                return 0
            # Count top-level comments (those without in_reply_to_id are conversation starters)
            unresolved = sum(1 for comment in comments if not comment.get('in_reply_to_id'))
            return unresolved
        except Exception:
            return 0

    def _load_disk_cache(self) -> Dict[str, Optional[str]]:
        """Load job logs cache from disk."""
        cache_file = self._cache_dir / "job_logs_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_disk_cache(self, cache: Dict[str, Optional[str]]) -> None:
        """Save job logs cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / "job_logs_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass  # Fail silently if we can't save cache

    def _save_to_disk_cache(self, job_id: str, error_summary: str) -> None:
        """Save a single job log to disk cache."""
        try:
            disk_cache = self._load_disk_cache()
            disk_cache[job_id] = error_summary
            self._save_disk_cache(disk_cache)
        except Exception:
            pass  # Fail silently if we can't save cache

    def get_required_checks(self, owner: str, repo: str, pr_number: int) -> set:
        """Get the list of required check names for a PR using gh CLI.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number

        Returns:
            Set of required check names
        """
        try:
            # Use gh CLI to get required checks (more reliable than API)
            import subprocess
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}', '--required'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Note: gh returns exit code 1 if there are failed checks, but still outputs the data
            # So we check stderr for actual errors, not return code
            if result.stderr and 'error' in result.stderr.lower():
                return set()

            # Parse output to extract check names
            # Format: "check_name\tstatus\tduration\turl"
            required_checks = set()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Split on tabs, get first column (check name)
                    parts = line.split('\t')
                    if parts:
                        check_name = parts[0].strip()
                        if check_name:
                            required_checks.add(check_name)

            return required_checks

        except Exception:
            # If we can't get required checks, return empty set
            return set()

    def _extract_pytest_summary(self, all_lines: list) -> Optional[str]:
        """Extract pytest short test summary from log lines.

        Args:
            all_lines: List of log lines

        Returns:
            Formatted summary with failed test names, or None if not a pytest failure
        """
        try:
            # Find the "short test summary info" section
            summary_start_idx = None
            summary_end_idx = None

            for i, line in enumerate(all_lines):
                if '=== short test summary info ===' in line or '==== short test summary info ====' in line:
                    summary_start_idx = i
                elif summary_start_idx is not None and '===' in line and 'failed' in line.lower() and 'passed' in line.lower():
                    # Found end line like "===== 4 failed, 329 passed, 3 skipped, 284 deselected in 422.68s (0:07:02) ====="
                    summary_end_idx = i
                    break

            if summary_start_idx is None:
                return None

            # Extract failed test lines
            failed_tests = []
            for i in range(summary_start_idx + 1, summary_end_idx if summary_end_idx else len(all_lines)):
                line = all_lines[i]
                # Look for lines that start with "FAILED" after timestamp/prefix
                if 'FAILED' in line:
                    # Extract the test name
                    # Format: "FAILED lib/bindings/python/tests/test_metrics_registry.py::test_counter_introspection"
                    parts = line.split('FAILED')
                    if len(parts) >= 2:
                        test_name = parts[1].strip()
                        # Remove any trailing info like " - AssertionError: ..."
                        if ' - ' in test_name:
                            test_name = test_name.split(' - ')[0].strip()
                        failed_tests.append(test_name)

            if not failed_tests:
                return None

            # Format the output
            result = "Failed unit tests:\n\n"
            for test in failed_tests:
                result += f"  • {test}\n"

            # Add summary line if available
            if summary_end_idx:
                summary_line = all_lines[summary_end_idx]
                # Extract just the summary part (after the timestamp and ====)
                # Format: "2025-11-04T20:25:55.7767823Z ==== 5 failed, 4 passed, 320 deselected, 111 warnings in 1637.48s (0:27:17) ===="
                if '====' in summary_line:
                    # Split by ==== and get the middle part
                    parts = summary_line.split('====')
                    if len(parts) >= 2:
                        summary_text = parts[1].strip()
                        if summary_text:
                            result += f"\nSummary: {summary_text}"

            return result

        except Exception:
            return None

    def get_job_error_summary(self, run_id: str, job_url: str, owner: str, repo: str) -> Optional[str]:
        """Get error summary by fetching job logs via gh CLI (with disk + memory caching).

        Args:
            run_id: Workflow run ID
            job_url: Job URL (contains job ID)
            owner: Repository owner
            repo: Repository name

        Returns:
            Error summary string or None
        """
        try:
            # Extract job ID from URL
            # Example: https://github.com/ai-dynamo/dynamo/actions/runs/18697156351/job/53317461976
            if '/job/' not in job_url:
                return None

            job_id = job_url.split('/job/')[1].split('?')[0]

            # Check in-memory cache first (fastest)
            if job_id in self._job_log_cache:
                return self._job_log_cache[job_id]

            # Check disk cache second
            disk_cache = self._load_disk_cache()
            if job_id in disk_cache:
                # Load into memory cache for faster subsequent access
                self._job_log_cache[job_id] = disk_cache[job_id]
                return disk_cache[job_id]

            # Fetch job logs via gh api (more reliable than gh run view)
            import subprocess
            result = subprocess.run(
                ['gh', 'api', f'/repos/{owner}/{repo}/actions/jobs/{job_id}/logs'],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode != 0 or not result.stdout:
                # If logs aren't available, return placeholder
                error_msg = result.stderr if result.stderr else "Unknown error"
                error_summary = f"Could not fetch job logs: {error_msg}\n\nView full logs at:\n{job_url}"
                self._job_log_cache[job_id] = error_summary
                self._save_to_disk_cache(job_id, error_summary)
                return error_summary

            # Get all log lines
            all_lines = result.stdout.strip().split('\n')

            # First, try to extract pytest short test summary (most useful for test failures)
            pytest_summary = self._extract_pytest_summary(all_lines)
            if pytest_summary:
                self._job_log_cache[job_id] = pytest_summary
                self._save_to_disk_cache(job_id, pytest_summary)
                return pytest_summary

            # Filter for error-related lines with surrounding context
            error_keywords = ['error', 'fail', 'Error', 'ERROR', 'FAIL', 'fatal', 'FATAL', 'broken']
            error_indices = []

            # Find all lines with error keywords
            for i, line in enumerate(all_lines):
                if line.strip() and not line.startswith('#'):
                    if any(keyword in line for keyword in error_keywords):
                        error_indices.append(i)

            # If we found error lines, extract them with surrounding context
            if error_indices:
                # For each error, get 10 lines before and 5 lines after
                context_lines = set()
                for error_idx in error_indices:
                    # Add lines before (up to 10)
                    for i in range(max(0, error_idx - 10), error_idx):
                        context_lines.add(i)
                    # Add error line itself
                    context_lines.add(error_idx)
                    # Add lines after (up to 5)
                    for i in range(error_idx + 1, min(len(all_lines), error_idx + 6)):
                        context_lines.add(i)

                # Sort indices and extract lines
                sorted_indices = sorted(context_lines)
                error_lines = []
                for idx in sorted_indices:
                    line = all_lines[idx]
                    if line.strip() and not line.startswith('#'):
                        error_lines.append(line)

                # Use up to last 80 lines with context (increased from 50)
                relevant_errors = error_lines[-80:]
                summary = '\n'.join(relevant_errors)

                # Limit length to 5000 chars (increased from 3000 for more context)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated, view full logs at job URL above)'

                self._job_log_cache[job_id] = summary
                self._save_to_disk_cache(job_id, summary)
                return summary

            # If no error keywords found, get last 40 lines as fallback
            last_lines = [line for line in all_lines if line.strip() and not line.startswith('#')][-40:]
            if last_lines:
                summary = '\n'.join(last_lines)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated)'
                self._job_log_cache[job_id] = summary
                self._save_to_disk_cache(job_id, summary)
                return summary

            error_summary = f"No error details found in logs.\n\nView full logs at:\n{job_url}"
            self._job_log_cache[job_id] = error_summary
            self._save_to_disk_cache(job_id, error_summary)
            return error_summary

        except subprocess.TimeoutExpired:
            error_summary = f"Log fetch timed out.\n\nView full logs at:\n{job_url}"
            self._job_log_cache[job_id] = error_summary
            self._save_to_disk_cache(job_id, error_summary)
            return error_summary

        except Exception as e:
            return f"Error fetching logs: {str(e)}\n\nView full logs at:\n{job_url}"

    def get_failed_checks(self, owner: str, repo: str, sha: str, required_checks: set, pr_number: Optional[int] = None) -> Tuple[List[FailedCheck], Optional[str]]:
        """Get failed CI checks for a commit using gh CLI.

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA (unused, kept for compatibility)
            required_checks: Set of required check names
            pr_number: PR number (optional, if available use gh pr checks)

        Returns:
            Tuple of (List of FailedCheck objects, rerun_url)
        """
        try:
            import subprocess

            # Use gh pr checks if PR number is available (more reliable)
            if pr_number:
                result = subprocess.run(
                    ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                # gh pr checks returns non-zero when there are failed checks
                # We only care if the command itself failed (no output)
                # So just check if we got output

                if not result.stdout.strip():
                    return [], None

                failed_checks = []
                rerun_run_id = None  # Track run_id for rerun command

                # Parse tab-separated output
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    check_name = parts[0]
                    status = parts[1]

                    # Only process failed checks
                    if status != 'fail':
                        continue

                    duration = parts[2] if len(parts) > 2 else ''
                    html_url = parts[3] if len(parts) > 3 else ''

                    # Extract run ID from html_url
                    check_run_id = ''
                    if '/runs/' in html_url:
                        check_run_id = html_url.split('/runs/')[1].split('/')[0]
                        # Save the first valid run_id we find for the rerun command
                        if not rerun_run_id and check_run_id:
                            rerun_run_id = check_run_id

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if check_run_id and html_url:
                        error_summary = self.get_job_error_summary(check_run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        run_id=check_run_id,
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

                # Sort: required checks first, then by name
                failed_checks.sort(key=lambda x: (not x.is_required, x.name))

                # Generate rerun URL if we have a run_id
                rerun_url = None
                if rerun_run_id:
                    rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{rerun_run_id}"

                return failed_checks, rerun_url

            # Fallback to GitHub API if no PR number
            endpoint = f"/repos/{owner}/{repo}/commits/{sha}/check-runs"
            data = self.get(endpoint)
            if not data or 'check_runs' not in data:
                return [], None

            failed_checks = []
            run_id = None

            for check in data['check_runs']:
                if check.get('conclusion') == 'failure':
                    check_name = check['name']

                    # Extract run ID from html_url
                    html_url = check.get('html_url', '')
                    if '/runs/' in html_url:
                        run_id = html_url.split('/runs/')[1].split('/')[0]

                    # Format duration
                    started = check.get('started_at', '')
                    completed = check.get('completed_at', '')
                    duration = ''
                    if started and completed:
                        try:
                            from datetime import datetime
                            start_time = datetime.fromisoformat(started.replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(completed.replace('Z', '+00:00'))
                            duration_sec = int((end_time - start_time).total_seconds())
                            if duration_sec >= 60:
                                duration = f"{duration_sec // 60}m{duration_sec % 60}s"
                            else:
                                duration = f"{duration_sec}s"
                        except Exception:
                            duration = "unknown"

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if run_id and html_url:
                        error_summary = self.get_job_error_summary(run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        run_id=run_id or '',
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

            # Sort: required checks first, then by name
            failed_checks.sort(key=lambda x: (not x.is_required, x.name))

            # Generate rerun URL if we have a run_id
            rerun_url = None
            if run_id:
                rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{run_id}"

            return failed_checks, rerun_url

        except Exception as e:
            import sys
            print(f"Error fetching failed checks: {e}", file=sys.stderr)
            return [], None

    def get_running_checks(self, pr_number: int, owner: str, repo: str, required_checks: set) -> List[RunningCheck]:
        """Get running CI checks for a PR using gh CLI.

        Args:
            pr_number: PR number
            owner: Repository owner
            repo: Repository name
            required_checks: Set of required check names

        Returns:
            List of RunningCheck objects
        """
        try:
            import subprocess
            result = subprocess.run(
                ['gh', 'pr', 'view', str(pr_number), '--repo', f'{owner}/{repo}', '--json', 'statusCheckRollup'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0 or not result.stdout:
                return []

            import json
            data = json.loads(result.stdout)
            checks = data.get('statusCheckRollup', [])

            running_checks = []
            for check in checks:
                status = check.get('status', '').upper()
                # Check if it's running
                if status in ('IN_PROGRESS', 'QUEUED', 'PENDING'):
                    name = check.get('name', '')
                    check_url = check.get('detailsUrl', '')
                    is_required = check.get('isRequired', False) or (name in required_checks)

                    # Calculate elapsed time
                    elapsed_time = None
                    started_at = check.get('startedAt')
                    if started_at and started_at != '0001-01-01T00:00:00Z':
                        try:
                            from datetime import datetime, timezone
                            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                            now = datetime.now(timezone.utc)
                            elapsed = now - start_time
                            total_seconds = int(elapsed.total_seconds())

                            if total_seconds < 60:
                                elapsed_time = f'{total_seconds}s'
                            else:
                                minutes = total_seconds // 60
                                seconds = total_seconds % 60
                                elapsed_time = f'{minutes}m{seconds}s'
                        except Exception:
                            elapsed_time = None
                    elif status == 'QUEUED':
                        elapsed_time = 'queued'

                    running_check = RunningCheck(
                        name=name,
                        check_url=check_url,
                        is_required=is_required,
                        elapsed_time=elapsed_time
                    )
                    running_checks.append(running_check)

            # Sort: required checks first, then by name
            running_checks.sort(key=lambda x: (not x.is_required, x.name))

            return running_checks

        except Exception as e:
            import sys
            print(f"Error fetching running checks for PR {pr_number}: {e}", file=sys.stderr)
            return []

    def get_pr_info(self, owner: str, repo: str, branch: str) -> List[PRInfo]:
        """Get PR information for a branch.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name

        Returns:
            List of PRInfo objects
        """
        endpoint = f"/repos/{owner}/{repo}/pulls"
        params = {'head': f'{owner}:{branch}', 'state': 'all'}

        try:
            prs_data = self.get(endpoint, params=params)
            if not prs_data:
                return []

            pr_list = []
            for pr_data in prs_data:
                # Fetch PR details in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Submit all 5 API calls in parallel
                    future_ci = executor.submit(self.get_ci_status, owner, repo, pr_data['head']['sha'], pr_data['number'])
                    future_conversations = executor.submit(self.count_unresolved_conversations, owner, repo, pr_data['number'])
                    future_details = executor.submit(self.get_pr_details, owner, repo, pr_data['number'])
                    future_required = executor.submit(self.get_required_checks, owner, repo, pr_data['number'])

                    # Wait for required checks first (needed for failed_checks)
                    required_checks = future_required.result()

                    # Now get failed checks and running checks with required info
                    future_failed_checks = executor.submit(self.get_failed_checks, owner, repo, pr_data['head']['sha'], required_checks, pr_data['number'])
                    future_running_checks = executor.submit(self.get_running_checks, pr_data['number'], owner, repo, required_checks)

                    # Wait for all to complete
                    ci_status = future_ci.result()
                    unresolved_count = future_conversations.result()
                    pr_details = future_details.result()
                    failed_checks, rerun_url = future_failed_checks.result()
                    running_checks = future_running_checks.result()

                mergeable = pr_details.get('mergeable') if pr_details else None
                mergeable_state = pr_details.get('mergeable_state') if pr_details else pr_data.get('mergeable_state', 'unknown')
                has_conflicts = (mergeable is False) or (mergeable_state == 'dirty')

                # Extract base branch for all PRs
                base_branch = pr_data.get('base', {}).get('ref', 'main')

                # Generate conflict message
                conflict_message = None
                if has_conflicts:
                    conflict_message = f"This branch has conflicts that must be resolved (merge {base_branch} into this branch)"

                # Generate blocking message based on mergeable_state
                blocking_message = None
                if mergeable_state in ['blocked', 'unstable', 'behind']:
                    if mergeable_state == 'unstable':
                        blocking_message = "Merging is blocked - Waiting on code owner review or required status checks"
                    elif mergeable_state == 'blocked':
                        blocking_message = "Merging is blocked - Required reviews or checks not satisfied"
                    elif mergeable_state == 'behind':
                        blocking_message = "This branch is out of date with the base branch"

                pr_info = PRInfo(
                    number=pr_data['number'],
                    title=pr_data['title'],
                    url=pr_data['html_url'],
                    state=pr_data['state'],
                    is_merged=pr_data.get('merged_at') is not None,
                    review_decision=pr_data.get('reviewDecision'),
                    mergeable_state=mergeable_state,
                    unresolved_conversations=unresolved_count,
                    ci_status=ci_status,
                    base_ref=base_branch,
                    created_at=pr_data.get('created_at'),
                    has_conflicts=has_conflicts,
                    conflict_message=conflict_message,
                    blocking_message=blocking_message,
                    failed_checks=failed_checks,
                    running_checks=running_checks,
                    rerun_url=rerun_url
                )
                pr_list.append(pr_info)

            return pr_list

        except Exception as e:
            import sys
            print(f"Error fetching PR info for {branch}: {e}", file=sys.stderr)
            return []


class GitLabAPIClient:
    """GitLab API client with automatic token detection and error handling.
    
    Features:
    - Automatic token detection (--token arg > GITLAB_TOKEN env > ~/.config/gitlab-token)
    - Request/response handling with proper error messages
    - Container registry queries
    
    Example:
        client = GitLabAPIClient()
        tags = client.get_registry_tags(project_id="169905", registry_id="85325")
    """
    
    @staticmethod
    def get_gitlab_token_from_file() -> Optional[str]:
        """Get GitLab token from ~/.config/gitlab-token file.
        
        Returns:
            GitLab token string, or None if not found
        """
        try:
            token_file = Path.home() / '.config' / 'gitlab-token'
            if token_file.exists():
                return token_file.read_text().strip()
        except Exception:
            pass
        return None
    
    def __init__(self, token: Optional[str] = None, base_url: str = "https://gitlab-master.nvidia.com"):
        """Initialize GitLab API client.
        
        Args:
            token: GitLab personal access token. If not provided, will try:
                   1. GITLAB_TOKEN environment variable
                   2. ~/.config/gitlab-token file
            base_url: GitLab instance URL (default: https://gitlab-master.nvidia.com)
        """
        # Token priority: 1) provided token, 2) environment variable, 3) config file
        self.token = token or os.environ.get('GITLAB_TOKEN') or self.get_gitlab_token_from_file()
        self.base_url = base_url.rstrip('/')
        self.headers = {}
        
        if self.token:
            self.headers['PRIVATE-TOKEN'] = self.token
    
    def has_token(self) -> bool:
        """Check if a GitLab token is configured."""
        return self.token is not None
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Any]:
        """Make GET request to GitLab API.
        
        Args:
            endpoint: API endpoint (e.g., "/api/v4/projects/169905")
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            JSON response (dict or list), or None if request failed
        """
        if not HAS_REQUESTS:
            # Fallback to urllib for basic GET requests
            import urllib.request
            import urllib.parse
            
            url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"
            
            if params:
                url += '?' + urllib.parse.urlencode(params)
            
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode())
            except Exception:
                return None
        
        # Use requests if available
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            
            if response.status_code == 401:
                raise Exception("GitLab API returned 401 Unauthorized. Check your token.")
            elif response.status_code == 403:
                raise Exception("GitLab API returned 403 Forbidden. Token may lack permissions.")
            elif response.status_code == 404:
                raise Exception(f"GitLab API returned 404 Not Found for {endpoint}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"GitLab API request failed for {endpoint}: {e}")
    
    def get_registry_tags(self, project_id: str, registry_id: str, 
                         per_page: int = 200, max_pages: int = 1) -> List[Dict[str, Any]]:
        """Get container registry tags for a project.
        
        Args:
            project_id: GitLab project ID (e.g., "169905")
            registry_id: Container registry ID (e.g., "85325")
            per_page: Number of tags per page (max 100 for GitLab)
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of tag dictionaries with 'name', 'location', etc.
        """
        if not self.has_token():
            return []
        
        all_tags = []
        
        for page in range(1, max_pages + 1):
            endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
            params = {
                'per_page': min(per_page, 100),  # GitLab API max is 100
                'page': page,
                'order_by': 'updated_at',
                'sort': 'desc'
            }
            
            try:
                tags = self.get(endpoint, params=params)
                if not tags:
                    break
                    
                all_tags.extend(tags)
                
                # If we got fewer tags than requested, we've reached the end
                if len(tags) < params['per_page']:
                    break
                    
            except Exception as e:
                if page == 1:
                    # Re-raise error on first page
                    raise
                # For subsequent pages, just stop
                break
        
        return all_tags
    
    def get_tag_details(self, project_id: str, registry_id: str, tag_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific tag.
        
        Args:
            project_id: GitLab project ID
            registry_id: Container registry ID
            tag_name: Tag name to get details for
            
        Returns:
            Tag details including total_size and created_at, or None if failed
        """
        if not self.has_token():
            return None
        
        try:
            # URL encode the tag name for the API
            import urllib.parse
            encoded_tag = urllib.parse.quote(tag_name, safe='')
            endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags/{encoded_tag}"
            return self.get(endpoint)
        except Exception:
            return None
    
    def get_cached_registry_tags(self, project_id: str, registry_id: str,
                                 cache_file: str = '.gitlab_tags_cache.json',
                                 max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Get registry tags with incremental page caching - stops when seeing duplicate pages.

        Args:
            project_id: GitLab project ID
            registry_id: Container registry ID
            cache_file: Path to cache file (default: .gitlab_tags_cache.json)
            max_age_hours: Maximum cache age in hours before refresh (default: 24)

        Returns:
            List of tag info dictionaries
        """
        import json
        from pathlib import Path
        from datetime import datetime, timedelta

        cache_path = Path(cache_file)
        tags = []
        cache_valid = False

        # Try to load from cache
        if cache_path.exists():
            try:
                cache_data = json.loads(cache_path.read_text())
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                cache_age = datetime.now() - cache_time

                if cache_age < timedelta(hours=max_age_hours):
                    tags = cache_data.get('tags', [])
                    cache_valid = True
            except Exception:
                pass

        # Fetch fresh data if cache is invalid
        if not cache_valid:
            # NOTE: GitLab Container Registry API does NOT support search/filter parameters
            # for listing tags. The API only returns tags in alphabetical order by name.
            # We must fetch ALL pages and cache them locally to enable searching by commit SHA.
            # Web UI search works via client-side filtering after fetching all tags.

            # Incremental fetch: page by page, stop when we see tags we already have
            tags = []
            seen_tag_names = set()
            page = 1
            safety_limit = 50  # Safety limit on pages (5000 tags total at 100 per page)
            per_page = 100

            while page <= safety_limit:
                # Fetch one page at a time using direct API call
                endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
                params = {
                    'per_page': per_page,
                    'page': page,
                    'order_by': 'updated_at',
                    'sort': 'desc'
                }

                try:
                    page_tags = self.get(endpoint, params=params)
                except Exception:
                    # API error, stop fetching
                    break

                if not page_tags:
                    # No more pages
                    break

                # Check if we've seen all tags on this page before
                page_tag_names = {t['name'] for t in page_tags}
                if page_tag_names.issubset(seen_tag_names):
                    # All tags on this page were already seen - we've reached the end
                    break

                # Add new tags
                new_tags = [t for t in page_tags if t['name'] not in seen_tag_names]
                tags.extend(new_tags)
                seen_tag_names.update(t['name'] for t in new_tags)

                # If we got fewer tags than requested, we've reached the end
                if len(page_tags) < per_page:
                    break

                page += 1

            # Save to cache
            try:
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'tags': tags,
                    'pages_fetched': page - 1
                }
                cache_path.write_text(json.dumps(cache_data, indent=2))
            except Exception:
                pass  # Silently fail on cache write errors

        return tags

    def get_registry_images_for_shas(self, project_id: str, registry_id: str,
                                     sha_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get container registry images for a list of commit SHAs.

        Args:
            project_id: GitLab project ID
            registry_id: Container registry ID
            sha_list: List of full commit SHAs (40 characters)

        Returns:
            Dictionary mapping SHA to list of image info dicts with:
                - tag: Full image tag
                - framework: Framework name (vllm/sglang/trtllm)
                - arch: Architecture (amd64/arm64)
                - pipeline_id: GitLab CI pipeline ID
                - location: Full image location URL
                - total_size: Image size in bytes
                - created_at: ISO 8601 timestamp
        """
        sha_to_images: Dict[str, List[Dict[str, Any]]] = {sha: [] for sha in sha_list}

        if not self.has_token():
            return sha_to_images

        try:
            # Get tags from cache (refreshes if older than 24 hours)
            tags = self.get_cached_registry_tags(project_id, registry_id)
            
            # Map tags to SHAs and fetch detailed info for each matching tag
            for tag_info in tags:
                tag_name = tag_info['name']
                
                # Tags format: {full-sha}-{pipeline-id}-{framework}-{arch}
                # Example: 02763376c536985cdd6a4f6402594f0cab328f21-38443994-vllm-amd64
                for sha_full in sha_list:
                    if sha_full in tag_name:
                        # Parse tag to extract framework and arch
                        # Format: SHA-PIPELINE-FRAMEWORK-ARCH
                        parts = tag_name.split('-')
                        if len(parts) >= 3:
                            # Last part is arch, second to last is framework
                            arch = parts[-1]
                            framework = parts[-2]
                            pipeline_id = parts[-3] if len(parts) >= 4 else "unknown"
                            
                            # Fetch detailed tag info (includes size and created_at)
                            tag_details = self.get_tag_details(project_id, registry_id, tag_name)
                            
                            if tag_details:
                                sha_to_images[sha_full].append({
                                    'tag': tag_name,
                                    'framework': framework,
                                    'arch': arch,
                                    'pipeline_id': pipeline_id,
                                    'location': tag_details.get('location', tag_info.get('location', '')),
                                    'total_size': tag_details.get('total_size', 0),
                                    'created_at': tag_details.get('created_at', ''),
                                    'registry_id': registry_id,
                                })
                            else:
                                # Fallback to basic info if details fetch fails
                                sha_to_images[sha_full].append({
                                    'tag': tag_name,
                                    'framework': framework,
                                    'arch': arch,
                                    'pipeline_id': pipeline_id,
                                    'location': tag_info.get('location', ''),
                                    'total_size': 0,
                                    'created_at': '',
                                    'registry_id': registry_id,
                                })
                            
        except Exception:
            # Silently fail - registry lookup is optional
            pass
        
        return sha_to_images


