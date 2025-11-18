#!/usr/bin/env python3
"""
Retag versioned dynamo local-dev Docker images to latest tags.

Example mappings:
  dynamo:v0.1.0.dev.ea07d51fc-vllm-local-dev   -> dynamo:latest-vllm-local-dev
  dynamo:v0.1.0.dev.ea07d51fc-sglang-local-dev -> dynamo:latest-sglang-local-dev
  dynamo:v0.1.0.dev.ea07d51fc-trtllm-local-dev -> dynamo:latest-trtllm-local-dev
"""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import common.py
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import docker
except ImportError:
    print("Error: docker package not found. Install with: pip install docker")
    sys.exit(1)

from common import FRAMEWORKS, get_framework_display_name, normalize_framework, BaseUtils, DockerUtils, DockerImageInfo


class DockerImageRetagger(BaseUtils):
    """Retag versioned Docker images to latest tags."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        super().__init__(dry_run=dry_run, verbose=verbose)
        self.docker_utils = DockerUtils(dry_run=dry_run, verbose=verbose)
        
    def find_retaggable_images(self, sha: Optional[str] = None) -> Dict[str, List[DockerImageInfo]]:
        """Find retaggable images grouped by framework, optionally filtered by SHA."""
        # Get dynamo local-dev images only
        all_images = self.docker_utils.list_dynamo_images(target="local-dev", sha=sha)
        
        # Group by framework
        retaggable = {framework: [] for framework in FRAMEWORKS}
        for image in all_images:
            if image.dynamo_info and image.dynamo_info.framework:
                retaggable[image.dynamo_info.framework].append(image)
                
        return retaggable
    
    def retag_framework_images(self, framework: str, images: List[DockerImageInfo], sha_filter: Optional[str] = None) -> int:
        """Retag images for framework. Returns success count."""
        if not images:
            self.logger.info(f"No {get_framework_display_name(framework)} local-dev images found to retag")
            return 0
            
        self.logger.info(f"\n=== Processing {get_framework_display_name(framework)} local-dev images ===")
        
        if sha_filter:
            # When SHA is specified, only retag the exact matching image
            matching_images = [img for img in images if img.matches_sha(sha_filter)]
            if len(matching_images) == 1:
                images_to_retag = matching_images
                self.logger.info(f"Retagging image with SHA: {sha_filter}")
            elif len(matching_images) == 0:
                self.logger.error(f"No {get_framework_display_name(framework)} local-dev images found with SHA '{sha_filter}'")
                self.logger.error(f"Available images: {[img.name for img in images]}")
                return 0
            else:
                self.logger.error(f"Found {len(matching_images)} {get_framework_display_name(framework)} local-dev images with SHA '{sha_filter}' - expected exactly 1")
                self.logger.error(f"Matching images: {[img.name for img in matching_images]}")
                self.logger.error("This suggests duplicate builds or SHA collision - please specify a more unique SHA")
                return 0
        else:
            # Default behavior: only retag newest image
            images_to_retag = [images[0]]  # Already sorted by creation date
            self.logger.info(f"Retagging newest image")
            
        success_count = 0
        for image in images_to_retag:
            if self.docker_utils.tag_image(image.name, image.get_latest_tag()):
                success_count += 1
                
        return success_count
    
    def run(self, frameworks: Optional[List[str]] = None, sha: Optional[str] = None) -> bool:
        """Run retagging process. Returns True if all successful."""
        self.logger.info("=== Docker Image Retagging Script ===")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No images will be retagged")
        
        if sha:
            self.logger.info(f"Source SHA filter: {sha}")
        else:
            self.logger.info("Source: newest images")
        
        retaggable_images = self.find_retaggable_images(sha)
        
        if frameworks:
            frameworks = [normalize_framework(f) for f in frameworks]
            retaggable_images = {k: v for k, v in retaggable_images.items() if k in frameworks}
            
        total_success = 0
        total_attempted = 0
        
        for framework, images in retaggable_images.items():
            if images:
                success_count = self.retag_framework_images(framework, images, sha_filter=sha)
                total_success += success_count
                # Always attempt to retag exactly one image per framework (newest or SHA-specific)
                total_attempted += 1
                
        self.logger.info(f"\n=== Summary ===")
        self.logger.info(f"Successfully retagged: {total_success}/{total_attempted} images")
        
        return total_success == total_attempted


def main():
    parser = argparse.ArgumentParser(
        description="Retag versioned dynamo local-dev Docker images to latest tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Retag newest local-dev images to latest-framework-local-dev
  %(prog)s --dry-run                 # Show what would be done
  %(prog)s --sha ea07d51fc           # Only retag local-dev images with this SHA
  %(prog)s --framework vllm sglang   # Only VLLM and SGLang local-dev images
        """
    )
    
    parser.add_argument(
        "--dry-run", "--dryrun",
        action="store_true",
        help="Show what would be done without executing commands"
    )
    
    parser.add_argument(
        "--sha",
        help="Filter source images by SHA (default: use newest images)"
    )
    
    parser.add_argument(
            "--framework", "--frameworks",
            nargs="*",
            choices=FRAMEWORKS,
            help="Specific frameworks to process (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    retagger = DockerImageRetagger(dry_run=args.dry_run, verbose=args.verbose)
    success = retagger.run(frameworks=args.framework, sha=args.sha)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
