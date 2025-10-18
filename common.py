"""
Dynamo utilities package.

Shared constants and utilities for dynamo Docker management scripts.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker
except ImportError:
    docker = None

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


# COMMENTED OUT: V2 does not use this. Kept for V1/retag script reference only.
# Forward declarations for type hints (methods below still use these types)
DynamoImageInfo = None
DockerImageInfo = None

# @dataclass
# class DynamoImageInfo:
#     """Dynamo-specific Docker image information.
#
#     DEPRECATION: retag script only. V1 and V2 do not use this.
#     """
#     version: Optional[str] = None      # Parsed version (e.g., "0.1.0.dev.ea07d51fc")
#     framework: Optional[str] = None    # Framework name (vllm, sglang, trtllm)
#     target: Optional[str] = None       # Target type (local-dev, dev, etc.)
#     latest_tag: Optional[str] = None   # Corresponding latest tag
#
#     def matches_sha(self, sha: str) -> bool:
#         """Check if this image matches the specified SHA."""
#         return self.version and sha in self.version
#
#     def is_framework_image(self) -> bool:
#         """Check if this has framework information."""
#         return self.framework is not None
#
#     def get_latest_tag(self, repository: str = "dynamo") -> str:
#         """Get the latest tag for this dynamo image."""
#         if self.latest_tag:
#             return self.latest_tag
#         if self.framework:
#             if self.target == "dev":
#                 # dev target maps to just latest-framework (no -dev suffix)
#                 return f"{repository}:latest-{self.framework}"
#             elif self.target:
#                 # other targets like local-dev keep the suffix
#                 return f"{repository}:latest-{self.framework}-{self.target}"
#             else:
#                 return f"{repository}:latest-{self.framework}"
#         return f"{repository}:latest"


# COMMENTED OUT: V2 does not use this. Kept for retag script reference only.
# @dataclass
# class DockerImageInfo:
#     """Comprehensive Docker image information.
#
#     DEPRECATION: retag script only. V1 and V2 do not use this.
#     """
#     name: str                    # Full image name (repo:tag)
#     repository: str              # Repository name
#     tag: str                     # Tag name
#     image_id: str               # Docker image ID
#     created_at: str             # Creation timestamp
#     size_bytes: int             # Size in bytes
#     size_human: str             # Human readable size
#     labels: Dict[str, str]      # Image labels
#
#     # Dynamo-specific information (optional)
#     dynamo_info: Optional[DynamoImageInfo] = None
#
#     def matches_sha(self, sha: str) -> bool:
#         """Check if this image matches the specified SHA."""
#         return self.dynamo_info and self.dynamo_info.matches_sha(sha)
#
#     def is_dynamo_image(self) -> bool:
#         """Check if this is a dynamo image."""
#         return self.repository in ["dynamo", "dynamo-base"]
#
#     def is_dynamo_framework_image(self) -> bool:
#         """Check if this is a dynamo framework image."""
#         return self.is_dynamo_image() and self.dynamo_info and self.dynamo_info.is_framework_image()
#
#     def get_latest_tag(self) -> str:
#         """Get the latest tag for this image."""
#         if self.dynamo_info:
#             return self.dynamo_info.get_latest_tag(self.repository)
#         return f"{self.repository}:latest"


def setup_logging(verbose: bool = False, name: str = None) -> logging.Logger:
    """Set up logging configuration for dynamo utilities.

    Args:
        verbose: Enable debug logging with timestamps
        name: Logger name (defaults to caller's __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()

    if verbose:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    else:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
                mock_result = subprocess.CompletedProcess(command, 0)
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
                        self.logger.debug(f"Composite SHA: {result}")
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
            sha: Composite SHA to store
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
        self.logger.info(f"Composite SHA file: {self.repo_path}/.last_build_composite_sha")

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
                    self.logger.info(f"Composite SHA unchanged ({current_sha[:12]}) but --force-run specified - proceeding")
                    return True
                else:
                    self.logger.info(f"Composite SHA unchanged ({current_sha[:12]}) - skipping rebuild")
                    self.logger.info("Use --force-run to force rebuild")
                    return False  # No rebuild needed
            else:
                self.logger.info("Composite SHA changed:")
                self.logger.info(f"  Previous: {stored_sha[:12]}")
                self.logger.info(f"  Current:  {current_sha[:12]}")
                self.logger.info("Rebuild needed")
                self.store_composite_sha(current_sha)
                return True
        else:
            self.logger.info(f"No previous composite SHA found - rebuild needed")
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
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
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
        normalized_framework = FrameworkInfo.normalize_framework(framework)
        if not FrameworkInfo.is_valid_framework(normalized_framework):
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
    
    def list_images(self, name_filter: str = None) -> List[DockerImageInfo]:
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
    
    def list_dynamo_images(self, framework: str = None, target: str = None, sha: str = None) -> List[DockerImageInfo]:
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
            framework = FrameworkInfo.normalize_framework(framework)
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
        import re
        
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


