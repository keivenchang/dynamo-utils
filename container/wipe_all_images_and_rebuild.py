#!/usr/bin/env python3
"""
Wipe All Images and Rebuild

⚠️  DANGER: This script will DELETE ALL Docker containers and images!

This script performs a complete Docker cleanup and rebuilds images for specified commit SHAs.
It can automatically determine which commits to build based on the last N image SHAs from
the git history.

Usage:
    # Build last 2 image SHAs
    python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --num-image-sha-to-build 2

    # Build specific commit SHAs
    python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --commit-sha 90ed9ab0e --commit-sha 34c4882d8

    # Dry run (show what would be done)
    python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --num-image-sha-to-build 2 --dry-run
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import common.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import git

from common import DynamoRepositoryUtils


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class DockerCleaner:
    """Handles dangerous Docker cleanup operations."""

    @staticmethod
    def stop_all_containers(dry_run: bool = False) -> bool:
        """Stop all running Docker containers."""
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}STEP 1: Stopping all containers{Colors.RESET}")
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

        if dry_run:
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would run: docker ps -aq | xargs docker stop{Colors.RESET}")
            return True

        try:
            # Get list of containers
            result = subprocess.run(
                ["docker", "ps", "-aq"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                # Stop containers
                subprocess.run(
                    ["docker", "stop"] + result.stdout.strip().split('\n'),
                    capture_output=True,
                    check=False
                )
                logger.info(f"{Colors.GREEN}✓ All containers stopped{Colors.RESET}")
            else:
                logger.info(f"{Colors.YELLOW}No containers to stop{Colors.RESET}")

            return True
        except Exception as e:
            logger.error(f"{Colors.RED}Error stopping containers: {e}{Colors.RESET}")
            return False

    @staticmethod
    def remove_all_containers(dry_run: bool = False) -> bool:
        """Remove all Docker containers."""
        logger.info(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}STEP 2: Removing all containers{Colors.RESET}")
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

        if dry_run:
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would run: docker ps -aq | xargs docker rm -f{Colors.RESET}")
            return True

        try:
            # Get list of containers
            result = subprocess.run(
                ["docker", "ps", "-aq"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                # Remove containers
                subprocess.run(
                    ["docker", "rm", "-f"] + result.stdout.strip().split('\n'),
                    capture_output=True,
                    check=False
                )
                logger.info(f"{Colors.GREEN}✓ All containers removed{Colors.RESET}")
            else:
                logger.info(f"{Colors.YELLOW}No containers to remove{Colors.RESET}")

            return True
        except Exception as e:
            logger.error(f"{Colors.RED}Error removing containers: {e}{Colors.RESET}")
            return False

    @staticmethod
    def remove_all_images(dry_run: bool = False) -> bool:
        """Force-remove all Docker images."""
        logger.info(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}STEP 3: Force-removing all Docker images{Colors.RESET}")
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

        if dry_run:
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would run: docker images -aq | xargs docker rmi -f{Colors.RESET}")
            return True

        try:
            # Get list of images
            result = subprocess.run(
                ["docker", "images", "-aq"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                # Remove images (this will produce a lot of output)
                logger.info(f"{Colors.YELLOW}Removing images (this may take a while)...{Colors.RESET}")
                subprocess.run(
                    ["docker", "rmi", "-f"] + result.stdout.strip().split('\n'),
                    check=False
                )
                logger.info(f"{Colors.GREEN}✓ All images removed{Colors.RESET}")
            else:
                logger.info(f"{Colors.YELLOW}No images to remove{Colors.RESET}")

            return True
        except Exception as e:
            logger.error(f"{Colors.RED}Error removing images: {e}{Colors.RESET}")
            return False

    @staticmethod
    def prune_system(dry_run: bool = False, include_volumes: bool = True) -> bool:
        """Prune Docker system."""
        logger.info(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}STEP 4: Pruning Docker system{Colors.RESET}")
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

        if dry_run:
            cmd = "docker system prune -af"
            if include_volumes:
                cmd += " --volumes"
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would run: {cmd}{Colors.RESET}")
            return True

        try:
            cmd = ["docker", "system", "prune", "-af"]
            if include_volumes:
                cmd.append("--volumes")

            subprocess.run(cmd, check=True)
            logger.info(f"{Colors.GREEN}✓ Docker system pruned{Colors.RESET}")
            return True
        except Exception as e:
            logger.error(f"{Colors.RED}Error pruning system: {e}{Colors.RESET}")
            return False


class ImageSHAResolver:
    """Resolves commit SHAs from image SHAs in git history."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo_utils = DynamoRepositoryUtils(str(repo_path))

        if git is None:
            raise RuntimeError("GitPython is required. Install with: pip install gitpython")

        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise RuntimeError(f"Not a git repository: {repo_path}")

    def get_last_n_image_shas(self, n: int) -> List[tuple[str, str]]:
        """
        Get the last N unique image SHAs from git history.
        
        Walks backward through commit history and records the LAST commit
        (chronologically oldest) with each unique image SHA before it changes.

        Returns:
            List of tuples: [(commit_sha, image_sha), ...]
        """
        logger.info(f"{Colors.CYAN}Analyzing git history to find last {n} unique image SHAs...{Colors.RESET}")

        # First pass: collect all commits with their image SHAs
        all_commits = []
        
        # Save current HEAD
        original_head = self.repo.head.commit.hexsha

        # Iterate through commits on main branch
        try:
            # Use main branch, not HEAD (which could be detached)
            ref = 'origin/main' if 'origin/main' in [str(r) for r in self.repo.refs] else 'main'
            for commit in self.repo.iter_commits(ref, max_count=500):
                commit_sha = commit.hexsha[:9]

                # Calculate image SHA for this commit by checking out the commit
                try:
                    # Checkout quietly
                    self.repo.git.checkout(commit.hexsha, force=True, quiet=True)
                    
                    # Compute image SHA
                    image_sha = self.repo_utils.generate_composite_sha()

                    if image_sha:
                        all_commits.append((commit_sha, image_sha))
                        
                except Exception as e:
                    logger.debug(f"  Skipping commit {commit_sha}: {e}")
                    continue

        finally:
            # Return to original HEAD
            try:
                self.repo.git.checkout(original_head, force=True, quiet=True)
                logger.debug(f"Returned to original HEAD: {original_head[:9]}")
            except Exception as e:
                logger.warning(f"{Colors.YELLOW}Failed to return to original HEAD: {e}{Colors.RESET}")
        
        # Second pass: find the last commit (chronologically oldest) with each unique image SHA
        results = []
        seen_image_shas = set()
        
        for i, (commit_sha, image_sha) in enumerate(all_commits):
            # Check if next commit has different image SHA (or if this is the last commit)
            is_last_with_this_sha = (
                i == len(all_commits) - 1 or  # Last commit overall
                all_commits[i + 1][1] != image_sha  # Next commit has different SHA
            )
            
            if is_last_with_this_sha and image_sha not in seen_image_shas:
                seen_image_shas.add(image_sha)
                results.append((commit_sha, image_sha[:7]))
                logger.info(f"  Found: commit {commit_sha} → image SHA {image_sha[:7]}")
                
                if len(results) >= n:
                    break

        if len(results) < n:
            logger.warning(f"{Colors.YELLOW}Only found {len(results)} unique image SHAs (requested {n}){Colors.RESET}")

        # Reverse to get chronological order (oldest first, newest last)
        # Since we walked commits newest→oldest, results are in reverse chronological order
        return list(reversed(results))


class ImageRebuilder:
    """Handles building images for specific commits."""

    def __init__(self, repo_path: Path, build_script: Path, email: Optional[str] = None):
        self.repo_path = repo_path
        self.build_script = build_script
        self.email = email

        if not self.build_script.exists():
            raise FileNotFoundError(f"Build script not found: {build_script}")

        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise RuntimeError(f"Not a git repository: {repo_path}")

    def build_for_sha(self, commit_sha: str, dry_run: bool = False) -> bool:
        """Build images for a specific commit SHA."""
        logger.info(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}Building images for SHA: {commit_sha}{Colors.RESET}")
        logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

        if dry_run:
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would checkout: {commit_sha}{Colors.RESET}")
            logger.info(f"{Colors.YELLOW}[DRY RUN] Would run build_images.py{Colors.RESET}")
            return True

        try:
            # Checkout the specific SHA
            logger.info(f"{Colors.YELLOW}Checking out SHA: {commit_sha}{Colors.RESET}")
            self.repo.git.checkout(commit_sha, force=True)

            # Show current commit info
            logger.info(f"{Colors.YELLOW}Current commit:{Colors.RESET}")
            logger.info(f"  {self.repo.head.commit.hexsha[:9]} {self.repo.head.commit.summary}")

            # Build command
            cmd = [
                "python3",
                str(self.build_script),
                "--repo-path", str(self.repo_path),
                "--parallel",
                "--run-ignore-lock",
            ]

            if self.email:
                cmd.extend(["--email", self.email])

            logger.info(f"{Colors.YELLOW}Starting build...{Colors.RESET}")
            logger.info(f"{Colors.CYAN}Command: {' '.join(cmd)}{Colors.RESET}")

            # Run build (don't use check=True since we want to continue even if build fails)
            result = subprocess.run(cmd)

            if result.returncode == 0:
                logger.info(f"{Colors.GREEN}✓ Build completed successfully for SHA: {commit_sha}{Colors.RESET}")
                return True
            else:
                logger.warning(f"{Colors.YELLOW}⚠ Build completed with errors for SHA: {commit_sha} (exit code: {result.returncode}){Colors.RESET}")
                return False

        except Exception as e:
            logger.error(f"{Colors.RED}Error building for SHA {commit_sha}: {e}{Colors.RESET}")
            return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="⚠️  DANGER: Wipe all Docker images and rebuild from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build last 2 image SHAs
  python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --num-image-sha-to-build 2

  # Build specific commit SHAs
  python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --commit-sha 90ed9ab0e --commit-sha 34c4882d8

  # Dry run
  python3 container/wipe_all_images_and_rebuild.py --repo-path ~/dynamo/dynamo_ci --num-image-sha-to-build 2 --dry-run
        """
    )

    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="Path to the Dynamo repository"
    )

    # Mutually exclusive: either specify number of image SHAs or specific commit SHAs
    sha_group = parser.add_mutually_exclusive_group(required=True)
    sha_group.add_argument(
        "--num-image-sha-to-build",
        type=int,
        help="Number of last image SHAs to build (determines commit SHAs automatically)"
    )
    sha_group.add_argument(
        "--commit-sha",
        action="append",
        dest="commit_shas",
        help="Specific commit SHA(s) to build (can specify multiple times)"
    )

    parser.add_argument(
        "--email",
        type=str,
        help="Email address for build notifications"
    )

    parser.add_argument(
        "--dry-run", "--dryrun",
        action="store_true",
        dest="dry_run",
        help="Show what would be done without executing"
    )

    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip Docker cleanup (only run builds)"
    )

    parser.add_argument(
        "--no-prune-volumes",
        action="store_true",
        help="Don't prune Docker volumes during cleanup"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Expand paths
    repo_path = args.repo_path.expanduser().resolve()
    build_script = Path(__file__).parent / "build_images.py"

    if not repo_path.exists():
        logger.error(f"{Colors.RED}Repository path does not exist: {repo_path}{Colors.RESET}")
        return 1

    # Warning message
    if not args.dry_run:
        logger.warning(f"{Colors.RED}⚠️  WARNING: This will DELETE ALL Docker containers and images!{Colors.RESET}")
        logger.info(f"{Colors.YELLOW}Press Ctrl+C within 5 seconds to cancel...{Colors.RESET}")
        time.sleep(5)

    # Step 1-4: Docker cleanup
    if not args.skip_cleanup:
        cleaner = DockerCleaner()

        if not cleaner.stop_all_containers(args.dry_run):
            logger.error(f"{Colors.RED}Failed to stop containers{Colors.RESET}")
            return 1

        if not cleaner.remove_all_containers(args.dry_run):
            logger.error(f"{Colors.RED}Failed to remove containers{Colors.RESET}")
            return 1

        if not cleaner.remove_all_images(args.dry_run):
            logger.error(f"{Colors.RED}Failed to remove images{Colors.RESET}")
            return 1

        if not cleaner.prune_system(args.dry_run, not args.no_prune_volumes):
            logger.error(f"{Colors.RED}Failed to prune system{Colors.RESET}")
            return 1

    # Determine which commit SHAs to build
    if args.num_image_sha_to_build:
        try:
            resolver = ImageSHAResolver(repo_path)
            sha_pairs = resolver.get_last_n_image_shas(args.num_image_sha_to_build)

            if not sha_pairs:
                logger.error(f"{Colors.RED}No image SHAs found in git history{Colors.RESET}")
                return 1

            commit_shas = [commit_sha for commit_sha, _ in sha_pairs]

            logger.info(f"\n{Colors.GREEN}{'=' * 60}{Colors.RESET}")
            logger.info(f"{Colors.GREEN}Resolved commit SHAs to build:{Colors.RESET}")
            for commit_sha, image_sha in sha_pairs:
                logger.info(f"  {commit_sha} (image SHA: {image_sha})")
            logger.info(f"{Colors.GREEN}{'=' * 60}{Colors.RESET}\n")

        except Exception as e:
            logger.error(f"{Colors.RED}Error resolving image SHAs: {e}{Colors.RESET}")
            return 1
    else:
        commit_shas = args.commit_shas

    # Build images for each SHA
    rebuilder = ImageRebuilder(repo_path, build_script, args.email)
    success_count = 0
    failure_count = 0

    for commit_sha in commit_shas:
        if rebuilder.build_for_sha(commit_sha, args.dry_run):
            success_count += 1
        else:
            failure_count += 1

    # Summary
    logger.info(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    logger.info(f"{Colors.BLUE}Build Summary{Colors.RESET}")
    logger.info(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    logger.info(f"{Colors.GREEN}✓ Successful builds: {success_count}{Colors.RESET}")
    if failure_count > 0:
        logger.info(f"{Colors.RED}✗ Failed builds: {failure_count}{Colors.RESET}")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
