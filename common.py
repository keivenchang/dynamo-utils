"""
Dynamo utilities package.

Shared constants and utilities for dynamo Docker management scripts.
"""

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker
except ImportError:
    docker = None

class FrameworkInfo:
    """Framework information and utilities for dynamo Docker images."""
    
    # Supported frameworks (canonical lowercase)
    FRAMEWORKS = ["vllm", "sglang", "trtllm"]
    
    # Framework display names
    FRAMEWORK_NAMES = {
        "vllm": "VLLM",
        "sglang": "SGLang", 
        "trtllm": "TensorRT-LLM"
    }
    
    @staticmethod
    def get_frameworks() -> List[str]:
        """Get list of supported frameworks (lowercase)."""
        return FrameworkInfo.FRAMEWORKS.copy()
    
    @staticmethod
    def get_frameworks_upper() -> List[str]:
        """Get list of supported frameworks (uppercase) - for backward compatibility."""
        return [f.upper() for f in FrameworkInfo.FRAMEWORKS]
    
    @staticmethod
    def normalize_framework(framework: str) -> str:
        """Normalize framework name to canonical lowercase form."""
        return framework.lower()
    
    @staticmethod
    def get_display_name(framework: str) -> str:
        """Get display name for framework (handles case conversion)."""
        normalized = FrameworkInfo.normalize_framework(framework)
        return FrameworkInfo.FRAMEWORK_NAMES.get(normalized, normalized.upper())
    
    @staticmethod
    def is_valid_framework(framework: str) -> bool:
        """Check if framework is supported."""
        return FrameworkInfo.normalize_framework(framework) in FrameworkInfo.FRAMEWORKS

# Backward compatibility exports
FRAMEWORKS = FrameworkInfo.get_frameworks()
FRAMEWORKS_UPPER = FrameworkInfo.get_frameworks_upper()
FRAMEWORK_NAMES = FrameworkInfo.FRAMEWORK_NAMES.copy()

# Backward compatibility wrappers (inline for simplicity)
normalize_framework = FrameworkInfo.normalize_framework
get_framework_display_name = FrameworkInfo.get_display_name


@dataclass
class DynamoImageInfo:
    """Dynamo-specific Docker image information."""
    version: Optional[str] = None      # Parsed version (e.g., "0.1.0.dev.ea07d51fc")
    framework: Optional[str] = None    # Framework name (vllm, sglang, trtllm)
    target: Optional[str] = None       # Target type (local-dev, dev, etc.)
    latest_tag: Optional[str] = None   # Corresponding latest tag
    
    def matches_sha(self, sha: str) -> bool:
        """Check if this image matches the specified SHA."""
        return self.version and sha in self.version
    
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
    """Comprehensive Docker image information."""
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
        return self.dynamo_info and self.dynamo_info.matches_sha(sha)
    
    def is_dynamo_image(self) -> bool:
        """Check if this is a dynamo image."""
        return self.repository in ["dynamo", "dynamo-base"]
    
    def is_dynamo_framework_image(self) -> bool:
        """Check if this is a dynamo framework image."""
        return self.is_dynamo_image() and self.dynamo_info and self.dynamo_info.is_framework_image()
    
    def get_latest_tag(self) -> str:
        """Get the latest tag for this image."""
        if self.dynamo_info:
            return self.dynamo_info.get_latest_tag(self.repository)
        return f"{self.repository}:latest"


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
        """Get comprehensive information about a Docker image."""
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
        """List dynamo framework images with optional filtering."""
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
        """Tag a Docker image."""
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
        """Retag multiple images to their latest tags."""
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
        """
        Remove unused --build-arg flags from Docker build commands for base images.
        
        Base images (dynamo-base) don't use most of the build arguments that are
        passed to framework-specific builds. Removing unused args helps Docker
        recognize when builds are truly identical.
        
        Args:
            docker_command: Docker build command string
            
        Returns:
            Filtered command string with unused build args removed
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
        """
        Normalize a Docker command by removing extra whitespace and sorting build args.
        This helps identify functionally identical commands with different formatting.
        
        Args:
            docker_command: Docker command string
            
        Returns:
            Normalized command string
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


