#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
devcontainer_sync.py - Development Configuration Synchronization Tool

PURPOSE:
  Automatically syncs development configuration files from a master location
  (dynamo-utils) to all Dynamo project directories, ensuring consistent
  development environments across multiple working copies using Jinja2 templates.

HOW IT WORKS:
  1. Reads devcontainer.json.j2 Jinja2 template
  2. Scans for directories matching 'dynamo*' pattern in parent directory
  3. Generates framework-specific devcontainer configs (VLLM, SGLANG, TRTLLM)
  4. Customizes each config with directory-specific names and tokens
  5. Tracks changes via MD5 hashes to avoid unnecessary syncs

USAGE:
  ./devcontainer_sync.py           # Normal sync operation
  ./devcontainer_sync.py --dryrun  # Preview changes without applying
  ./devcontainer_sync.py --force   # Force sync even if no changes detected
  ./devcontainer_sync.py --silent  # No output (for cron jobs)
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: jinja2 module not found. Install with: pip install jinja2")
    sys.exit(1)


class DevContainerSync:
    """Handles synchronization of devcontainer configurations."""

    def __init__(self, args):
        self.dryrun = args.dryrun
        self.force = args.force
        self.silent = args.silent
        
        # Get script directory and setup paths
        self.script_dir = Path(__file__).parent.resolve()
        self.template_file = self.script_dir / "devcontainer.json.j2"
        
        # Destination directory pattern
        dest_base = os.environ.get("DEVCONTAINER_SRC_DIR", f"{os.path.expanduser('~')}/nvidia/dynamo")
        self.dest_pattern = Path(dest_base).parent
        self.dest_prefix = Path(dest_base).name
        
        # Frameworks to generate
        self.frameworks = ["vllm", "sglang", "trtllm"]
        
        # Get username for customization
        self.username = os.environ.get("USER", "user")
        
        # Hash tracking
        self.temp_sha_file = Path("/tmp/.sync_devcontainer.sha")
        
    def log(self, message, prefix="INFO"):
        """Log a message unless in silent mode."""
        if self.silent:
            return
        
        if self.dryrun and prefix != "ERROR":
            print(f"[DRYRUN] {message}")
        else:
            print(f"{prefix}: {message}")
    
    def get_template_hash(self):
        """Calculate MD5 hash of the template file."""
        if not self.template_file.exists():
            self.log(f"ERROR: Template file not found at {self.template_file}", "ERROR")
            sys.exit(1)
        
        with open(self.template_file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_previous_hash(self):
        """Get the previously stored hash."""
        if self.temp_sha_file.exists():
            return self.temp_sha_file.read_text().strip()
        return None
    
    def save_hash(self, hash_value):
        """Save the current hash."""
        if not self.dryrun:
            self.temp_sha_file.write_text(hash_value)
    
    def find_target_directories(self):
        """Find all dynamo* directories to sync."""
        directories = []
        
        # Find all matching directories
        for item in self.dest_pattern.glob(f"{self.dest_prefix}*"):
            if item.is_dir() and item != self.script_dir:
                directories.append(item)
        
        return sorted(directories)
    
    def remove_comments(self, content):
        """Remove // comments from JSON content while preserving strings."""
        lines = []
        for line in content.split('\n'):
            # Skip lines that are entirely commented out (including those with leading //)
            stripped = line.lstrip()
            if stripped.startswith('//'):
                continue

            # Remove inline comments (// at end of line)
            # But preserve // inside strings by only removing // after the last quote pair
            # Simple approach: remove // and everything after it if not inside a string
            in_string = False
            escape = False
            comment_start = -1

            for i, char in enumerate(line):
                if escape:
                    escape = False
                    continue

                if char == '\\':
                    escape = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    comment_start = i
                    break

            if comment_start >= 0:
                line = line[:comment_start].rstrip()

            # Only add non-empty lines or lines that are just whitespace but preserve structure
            if line or not stripped:
                lines.append(line)

        return '\n'.join(lines)

    def render_template(self, framework, dirname, tokens=None):
        """Render the Jinja2 template for a specific framework and directory."""
        # Setup Jinja2 environment
        env = Environment(loader=FileSystemLoader(self.script_dir))
        template = env.get_template(self.template_file.name)

        # Get current year
        current_year = datetime.now().year

        # Prepare template variables
        template_vars = {
            "framework": framework,
            "current_year": current_year,
            "dirname": dirname,
            "username": self.username,
        }

        # Render the template
        rendered = template.render(**template_vars)

        # Remove comments from rendered output
        rendered = self.remove_comments(rendered)

        # Customize the rendered output
        # 1. Update the name field to include directory and username
        display_name = f"[{dirname}-{self.username}] {framework.upper()}"
        rendered = rendered.replace(
            f'"name": "Dynamo {framework.upper()} Dev Container"',
            f'"name": "{display_name}"'
        )

        # 2. Replace token placeholders if provided
        if tokens:
            github_token = tokens.get("GITHUB_TOKEN", "")
            hf_token = tokens.get("HF_TOKEN", "")
            rendered = rendered.replace("__GITHUB_TOKEN__", github_token)
            rendered = rendered.replace("__HF_TOKEN__", hf_token)

        return rendered
    
    def sync_directory(self, target_dir):
        """Sync devcontainer configs to a single directory."""
        dirname = target_dir.name
        self.log(f"Updating {target_dir}/...")
        
        # Get tokens from environment
        tokens = {
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        }
        
        # Generate configs for each framework
        for framework in self.frameworks:
            # Create target directory path
            framework_dir = target_dir / ".devcontainer" / f"{self.username}_{framework}"
            target_file = framework_dir / "devcontainer.json"
            
            # Create directory if it doesn't exist
            if not self.dryrun:
                framework_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.log(f"mkdir -p {framework_dir}")
            
            # Render template
            rendered_content = self.render_template(framework, dirname, tokens)
            
            # Write the file
            if not self.dryrun:
                target_file.write_text(rendered_content)
            
            if not self.silent:
                if self.dryrun:
                    self.log(f"+ cp (rendered template) {target_file}")
                else:
                    print(f"+ mkdir -p {framework_dir}")
                    print(f"+ cp (rendered template) {target_file}")
    
    def run(self):
        """Main synchronization logic."""
        # Check if template file exists
        if not self.template_file.exists():
            self.log(f"ERROR: Template file not found at {self.template_file}", "ERROR")
            return 1
        
        # Calculate current hash
        current_hash = self.get_template_hash()
        previous_hash = self.get_previous_hash()
        
        # Check if sync is needed
        if current_hash == previous_hash and not self.force:
            self.log("Development config files unchanged, no sync needed.", "DEBUG")
            return 0
        
        if self.force:
            self.log("Force flag detected, syncing regardless of changes...")
        
        self.log("Development config files have changed, syncing to subdirectories...")
        
        # Find target directories
        target_dirs = self.find_target_directories()
        
        if not target_dirs:
            self.log(f"No directories found matching {self.dest_pattern}/{self.dest_prefix}*")
            return 0
        
        # Skip the source directory
        if self.script_dir in target_dirs:
            self.log(f"Skipping source directory {self.script_dir}")
            target_dirs.remove(self.script_dir)
        
        # Sync each directory
        sync_count = 0
        for target_dir in target_dirs:
            self.sync_directory(target_dir)
            sync_count += 1
        
        # Save the new hash
        self.save_hash(current_hash)
        self.log(f"Sync completed. Updated {sync_count} directories.")
        
        return 0


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sync devcontainer configurations using Jinja2 templates"
    )
    parser.add_argument(
        "--dryrun", "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if no changes detected"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="No output (for cron jobs)"
    )
    
    args = parser.parse_args()
    
    syncer = DevContainerSync(args)
    return syncer.run()


if __name__ == "__main__":
    sys.exit(main())

