#!/usr/bin/env python3
"""
Comprehensive dynamo package checker, tester, and usage guide.

Combines version checking, import testing, and usage examples into a single tool.
Features dynamic component discovery and comprehensive troubleshooting guidance.

Usage:
    ./bin/dynamo_check.py                        # Run all checks
    ./bin/dynamo_check.py --imports              # Only test imports
    ./bin/dynamo_check.py --examples             # Only show examples
    ./bin/dynamo_check.py --try-pythonpath      # Test imports with workspace paths
    ./bin/dynamo_check.py --help                 # Show help
"""

import sys
import os
import argparse
import asyncio
import importlib.metadata
import logging
import subprocess
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False


class DynamoChecker:
    """Comprehensive dynamo package checker."""

    def __init__(self):
        self.workspace_dir = self._find_workspace()
        self.results = {}
        self._suppress_planner_warnings()

    def _suppress_planner_warnings(self):
        """Suppress Prometheus endpoint warnings from planner module during import testing."""
        # The planner module logs a warning about Prometheus endpoint when imported
        # outside of a Kubernetes cluster. Suppress this for cleaner output.
        planner_logger = logging.getLogger('dynamo.planner.defaults')
        planner_logger.setLevel(logging.ERROR)

    # ====================================================================
    # WORKSPACE AND COMPONENT DISCOVERY
    # ====================================================================

    def _find_workspace(self) -> str:
        """Find dynamo workspace directory.

        Returns:
            Path to workspace directory or empty string if not found
            Example: '.' (if current dir), '~/dynamo', '/workspace', or ''

        Note: Checks local path first, then common locations. Validates by looking for README.md file.
        """
        candidates = [
            ".",  # Current directory (local path)
            os.path.expanduser("~/dynamo"),
            "/workspace",
            "~/dynamo"
        ]

        for candidate in candidates:
            if self._is_dynamo_workspace(candidate):
                return candidate
        return ""

    def _is_dynamo_workspace(self, path: str) -> bool:
        """Check if a directory is a dynamo workspace by looking for characteristic files/directories.

        Args:
            path: Directory path to check

        Returns:
            True if directory appears to be a dynamo workspace

        Note: Checks for multiple indicators like README.md, components/, lib/bindings/, lib/runtime/, Cargo.toml, etc.
        """
        if not os.path.exists(path):
            return False

        # Check for characteristic dynamo workspace files and directories
        indicators = [
            "README.md",
            "components",
            "lib/bindings/python",
            "lib/runtime",
            "Cargo.toml"
        ]

        # Require at least 3 indicators to be confident it's a dynamo workspace
        found_indicators = 0
        for indicator in indicators:
            if os.path.exists(os.path.join(path, indicator)):
                found_indicators += 1

        return found_indicators >= 4

    def _discover_runtime_components(self) -> List[str]:
        """Discover ai-dynamo-runtime components from filesystem.

        Returns:
            List of runtime component module names
            Example: ['dynamo._core', 'dynamo.nixl_connect', 'dynamo.llm', 'dynamo.runtime']

        Note: Always includes 'dynamo._core' (compiled Rust module), then scans
              lib/bindings/python/src/dynamo/ for additional components.
        """
        components = ["dynamo._core"]  # Always include compiled Rust module

        if not self.workspace_dir:
            return components

        # Scan runtime components (llm, runtime, nixl_connect, etc.)
        # Examples: lib/bindings/python/src/dynamo/{llm,runtime,nixl_connect}/__init__.py
        runtime_path = f"{self.workspace_dir}/lib/bindings/python/src/dynamo"
        if not os.path.exists(runtime_path):
            print(f"⚠️  Warning: Runtime components directory not found: {runtime_path}")
            return components

        for item in os.listdir(runtime_path):
            item_path = os.path.join(runtime_path, item)
            if os.path.isdir(item_path) and os.path.exists(f"{item_path}/__init__.py"):
                components.append(f"dynamo.{item}")

        return components

    def _discover_framework_components(self) -> List[str]:
        """Discover ai-dynamo framework components from filesystem.

        Returns:
            List of framework component module names
            Example: ['dynamo.frontend', 'dynamo.planner', 'dynamo.vllm', 'dynamo.sglang', 'dynamo.llama_cpp']

        Note: Scans components/ and components/backends/ directories for modules with __init__.py files.
        """
        components = []

        if not self.workspace_dir:
            return components

        # Scan direct components (frontend, planner, etc.)
        # Examples: components/{frontend,planner}/src/dynamo/{frontend,planner}/__init__.py
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                item_path = os.path.join(comp_path, item)
                if (os.path.isdir(item_path) and
                    os.path.exists(f"{item_path}/src/dynamo/{item}/__init__.py")):
                    components.append(f"dynamo.{item}")
        else:
            print(f"⚠️  Warning: Components directory not found: {comp_path}")

        # Scan backend components (vllm, sglang, etc.)
        # Examples: components/backends/{vllm,sglang,llama_cpp}/src/dynamo/{vllm,sglang,llama_cpp}/__init__.py
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                item_path = os.path.join(backend_path, item)
                if (os.path.isdir(item_path) and
                    os.path.exists(f"{item_path}/src/dynamo/{item}/__init__.py")):
                    components.append(f"dynamo.{item}")
        else:
            print(f"⚠️  Warning: Backend components directory not found: {backend_path}")

        return components

    def _is_dynamo_build_available(self) -> bool:
        """Check if dynamo_build.sh is available in the same directory as this script.

        Returns:
            True if dynamo_build.sh exists in the same directory as dynamo_check.py
        """
        script_dir = Path(__file__).parent
        dynamo_build_path = script_dir / "dynamo_build.sh"
        return dynamo_build_path.exists()

    def _format_timestamp_pdt(self, timestamp: float) -> str:
        """Format a timestamp in PDT timezone.

        Args:
            timestamp: Unix timestamp

        Returns:
            Formatted timestamp string in PDT or local timezone
            Example: '2025-08-10 22:22:52 PDT'
        """
        if PYTZ_AVAILABLE:
            try:
                pdt = pytz.timezone('US/Pacific')
                dt = datetime.datetime.fromtimestamp(timestamp, tz=pdt)
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                # Fallback to UTC if PDT conversion fails
                try:
                    dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
                    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass

        # Fallback to local time with manual PDT offset approximation
        # PDT is UTC-7, so subtract 7 hours from UTC
        dt_utc = datetime.datetime.utcfromtimestamp(timestamp)
        dt_pdt = dt_utc - datetime.timedelta(hours=7)
        return dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")

    def _get_cargo_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get cargo target directory and cargo home directory.

        Returns:
            Tuple of (target_directory, cargo_home) or (None, None) if cargo not available
            Example: ('~/dynamo/.build/target', '/home/ubuntu/.cargo')
        """
        # First check if cargo is available
        try:
            subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
        except FileNotFoundError:
            print("⚠️  Warning: cargo command not found. Install Rust toolchain to see cargo target directory.")
            return None, None
        except subprocess.TimeoutExpired:
            print("⚠️  Warning: cargo command timed out")
            return None, None

        # Get cargo home directory
        cargo_home = os.environ.get("CARGO_HOME")
        if not cargo_home:
            cargo_home = os.path.expanduser("~/.cargo")

        # Get cargo target directory
        target_directory = None
        try:
            # Run cargo metadata command to get target directory
            result = subprocess.run(
                ["cargo", "metadata", "--format-version=1", "--no-deps"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.workspace_dir if self.workspace_dir else None
            )

            if result.returncode == 0:
                # Parse JSON output to extract target_directory
                import json
                metadata = json.loads(result.stdout)
                target_directory = metadata.get("target_directory")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                json.JSONDecodeError):
            # cargo metadata failed or JSON parsing failed
            pass

        return target_directory, cargo_home

    def _find_so_file(self, target_directory: str) -> Optional[str]:
        """Find the compiled *.so file in target directory or Python bindings.

        Args:
            target_directory: Path to cargo target directory

        Returns:
            Path to *.so file or None if not found
            Example: '~/dynamo/target/debug/libdynamo_core.so'
        """
        if not target_directory or not os.path.exists(target_directory):
            return None

        # Look for *.so files in debug and release directories
        for profile in ["debug", "release"]:
            profile_dir = os.path.join(target_directory, profile)
            if os.path.exists(profile_dir):
                try:
                    for root, dirs, files in os.walk(profile_dir):
                        for file in files:
                            if file.endswith(".so"):
                                return os.path.join(root, file)
                except OSError:
                    continue

        # Also check Python bindings directory for installed *.so
        if self.workspace_dir:
            bindings_dir = f"{self.workspace_dir}/lib/bindings/python/src/dynamo"
            if os.path.exists(bindings_dir):
                try:
                    for root, dirs, files in os.walk(bindings_dir):
                        for file in files:
                            if file.endswith(".so") and "_core" in file:
                                return os.path.join(root, file)
                except OSError:
                    pass

        return None

    def _get_cargo_build_profile(self, target_directory: str) -> Optional[str]:
        """Determine which cargo build profile (debug/release) was used most recently.

        Args:
            target_directory: Path to cargo target directory

        Returns:
            'debug', 'release', 'debug/release', or None if cannot determine
            Example: 'debug'
        """
        # First check environment variables that indicate current build profile
        profile_env = os.environ.get("PROFILE")
        if profile_env:
            if profile_env == "dev":
                return "debug"
            elif profile_env == "release":
                return "release"

        # Check OPT_LEVEL as secondary indicator
        opt_level = os.environ.get("OPT_LEVEL")
        if opt_level:
            if opt_level == "0":
                return "debug"
            elif opt_level in ["2", "3"]:
                return "release"

        # Fall back to filesystem inspection
        if not target_directory or not os.path.exists(target_directory):
            return None

        debug_dir = os.path.join(target_directory, "debug")
        release_dir = os.path.join(target_directory, "release")

        debug_exists = os.path.exists(debug_dir)
        release_exists = os.path.exists(release_dir)

        if not debug_exists and not release_exists:
            return None
        elif debug_exists and not release_exists:
            return "debug"
        elif release_exists and not debug_exists:
            return "release"
        else:
            # Both exist, check which was modified more recently
            try:
                debug_mtime = os.path.getmtime(debug_dir)
                release_mtime = os.path.getmtime(release_dir)

                if abs(debug_mtime - release_mtime) < 1.0:  # Same timestamp (within 1 second)
                    return "debug/release"  # Both available, runtime choice depends on invocation
                else:
                    return "release" if release_mtime > debug_mtime else "debug"
            except OSError:
                return None

    def _setup_pythonpath(self):
        """Set up PYTHONPATH for component imports."""
        if not self.workspace_dir:
            return

        paths = []

        # Collect component source paths
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                src_path = f"{comp_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)
        else:
            print(f"⚠️  Warning: Components directory not found for PYTHONPATH setup: {comp_path}")

        # Collect backend source paths
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                src_path = f"{backend_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)
        else:
            print(f"⚠️  Warning: Backend components directory not found for PYTHONPATH setup: {backend_path}")

        # Update sys.path for current process
        if paths:
            # Add paths to sys.path for immediate effect on imports
            for path in paths:
                if path not in sys.path:
                    sys.path.insert(0, path)  # Insert at beginning for priority

            # Show what PYTHONPATH would be (for manual shell setup)
            pythonpath_value = ":".join(paths)
            current_path = os.environ.get("PYTHONPATH", "")
            if current_path:
                pythonpath_value = f"{pythonpath_value}:{current_path}"

            print(f"Below are the results if you export PYTHONPATH=\"{pythonpath_value}\":")
            print(f"   ({len(paths)} workspace component paths found)")
            for path in paths:
                print(f"   • {path}")
            print()
        else:
            print("⚠️  Warning: No component source paths found for PYTHONPATH setup")

    # ====================================================================
    # IMPORT TESTING
    # ====================================================================

    def _test_component_group(self, components: List[str], package_name: str,
                             group_name: str, max_width: int, site_packages: str,
                             collect_failures: bool = False) -> Tuple[Dict[str, str], List[str]]:
        """Test a group of components for a given package.

        Args:
            components: List of component names to test
                Example: ['dynamo._core', 'dynamo.llm', 'dynamo.runtime']
            package_name: Name of the package to get version from
                Example: 'ai-dynamo-runtime'
            group_name: Display name for the group
                Example: 'Runtime components'
            max_width: Maximum width for component name alignment
                Example: 20
            site_packages: Path to site-packages directory
                Example: '/opt/dynamo/venv/lib/python3.12/site-packages'
            collect_failures: Whether to collect failed component names
                Example: True (for framework components), False (for runtime)

        Returns:
            Tuple of (results dict, list of failed components)
            Example: ({'dynamo._core': '✅ Success', 'dynamo.llm': '❌ Failed: No module named dynamo.llm'},
                     ['dynamo.llm'])

        Output printed to console:
            Runtime components (ai-dynamo-runtime 0.4.0):
               ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               ❌ dynamo.llm          No module named 'dynamo.llm'
        """
        results = {}
        failures = []

        # Print header with version info
        try:
            version = importlib.metadata.version(package_name)
            header = f"{group_name} ({package_name} {version}):"
        except importlib.metadata.PackageNotFoundError:
            header = f"{group_name} ({package_name} - Not installed):"
        except Exception:
            header = f"{group_name} ({package_name}):"

        # Add newline prefix for framework components
        if group_name.lower().startswith('framework'):
            header = f"\n{header}"

        print(header)

        # Test each component
        for component in components:
            try:
                module = __import__(component, fromlist=[''])
                results[component] = "✅ Success"
                # Get module path for location info
                module_path = getattr(module, '__file__', 'built-in')
                if module_path and module_path != 'built-in':
                    # Only show timestamps for generated files (*.so, *.pth, etc.), not __init__.py
                    timestamp_str = ""
                    show_timestamp = False

                    # Check if this is a generated file we want to show timestamps for
                    if any(module_path.endswith(ext) for ext in ['.so', '.pth', '.dll', '.dylib']):
                        show_timestamp = True

                    if show_timestamp:
                        try:
                            if os.path.exists(module_path):
                                mtime = os.path.getmtime(module_path)
                                timestamp_str = f" (modified: {self._format_timestamp_pdt(mtime)})"
                        except OSError:
                            pass

                    if self.workspace_dir and module_path.startswith(self.workspace_dir):
                        # From workspace source
                        rel_path = os.path.relpath(module_path, self.workspace_dir)
                        if show_timestamp:
                            print(f"   ✅ {component:<{max_width}} {rel_path}{timestamp_str}")
                        else:
                            print(f"   ✅ {component:<{max_width}} {rel_path}")
                    elif site_packages and module_path.startswith(site_packages):
                        # From installed package - show full path
                        if show_timestamp:
                            print(f"   ✅ {component:<{max_width}} {module_path}{timestamp_str}")
                        else:
                            print(f"   ✅ {component:<{max_width}} {module_path}")
                    else:
                        # Other location
                        if show_timestamp:
                            print(f"   ✅ {component:<{max_width}} {module_path}{timestamp_str}")
                        else:
                            print(f"   ✅ {component:<{max_width}} {module_path}")
                else:
                    built_in_suffix = " (built-in)" if group_name.lower().startswith('framework') else " built-in"
                    print(f"   ✅ {component:<{max_width}}{built_in_suffix}")
            except ImportError as e:
                results[component] = f"❌ Failed: {e}"
                print(f"   ❌ {component:<{max_width}} {e}")
                if collect_failures:
                    failures.append(component)

        return results, failures

    def test_imports(self) -> Dict[str, str]:
        """Test imports for all discovered components.

        Returns:
            Dictionary mapping component names to their import status
            Example: {
                'dynamo._core': '✅ Success',
                'dynamo.llm': '✅ Success',
                'dynamo.runtime': '✅ Success',
                'dynamo.frontend': '❌ Failed: No module named dynamo.frontend',
                'dynamo.planner': '✅ Success'
            }

        Console output example:
            Runtime components (ai-dynamo-runtime 0.4.0):
               ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               ✅ dynamo.llm          /opt/dynamo/venv/lib/.../dynamo/llm/__init__.py

            Framework components (ai-dynamo 0.4.0):
               ✅ dynamo.frontend     /opt/dynamo/venv/lib/.../dynamo/frontend/__init__.py
               ❌ dynamo.missing      No module named 'dynamo.missing'
        """
        results = {}

        # Show relevant .pth files and Python package info FIRST
        # so users see installation status before import checks
        self._show_relevant_pth_files()
        # Add a blank line before component import results for readability
        print()

        # Discover all components
        runtime_components = self._discover_runtime_components()
        framework_components = self._discover_framework_components()

        # Calculate max width for alignment across ALL components
        all_components = runtime_components + framework_components
        max_width = max(len(comp) for comp in all_components) if all_components else 0

        # Get site-packages path for comparison
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else None

        # NOTE: Installation dates are not available because:
        # - pip and other package managers don't store installation timestamps in metadata
        # - importlib.metadata only provides standard package info (version, author, etc.)
        # - File system timestamps are unreliable (reflect file creation, not installation time)
        # - No standard method exists across different package managers (pip, conda, etc.)

        # Test runtime components
        runtime_results, _ = self._test_component_group(
            runtime_components, 'ai-dynamo-runtime', 'Runtime components',
            max_width, site_packages, collect_failures=False
        )
        results.update(runtime_results)

        # Test framework components
        framework_results, framework_failures = self._test_component_group(
            framework_components, 'ai-dynamo', 'Framework components',
            max_width, site_packages, collect_failures=True
        )
        results.update(framework_results)

        # (Package info already shown at the top of this method)

        # Show PYTHONPATH recommendation if any framework components failed
        if framework_failures and self.workspace_dir:
            pythonpath = self._get_pythonpath()
            if pythonpath:
                print("\nMissing framework components detected. To resolve this, choose one of the following options:")
                print("1. For local development, set the PYTHONPATH environment variable:")
                print(f"   ./bin/dynamo_check.py --try-pythonpath --imports\n   export PYTHONPATH=\"{pythonpath}\"")
                not_found_suffix = "" if self._is_dynamo_build_available() else "  # (dynamo_build.sh not found)"
                print("2. For a production-release (slower build time), build the packages with:")
                print(f"   dynamo_build.sh --release{not_found_suffix}")

        # Show Rust cargo information (moved to bottom)
        cargo_target, cargo_home = self._get_cargo_info()
        if cargo_target or cargo_home:
            print()  # Add blank line before cargo info

            if cargo_home:
                cargo_home_env = os.environ.get("CARGO_HOME")
                if cargo_home_env:
                    print(f"Cargo home directory: {cargo_home} (CARGO_HOME is set)")
                else:
                    print(f"Cargo home directory: {cargo_home}")

            if cargo_target:
                cargo_target_env = os.environ.get("CARGO_TARGET_DIR")
                build_profile = self._get_cargo_build_profile(cargo_target)

                # Build the target directory message
                if cargo_target_env:
                    target_msg = f"Cargo target directory: {cargo_target} (CARGO_TARGET_DIR is set)"
                else:
                    target_msg = f"Cargo target directory: {cargo_target}"

                # Build profile information is shown in the debug/release directory details below
                # No need to show it in the main target directory line

                print(target_msg)

                # Show debug and release directories on separate lines
                debug_dir = os.path.join(cargo_target, "debug")
                release_dir = os.path.join(cargo_target, "release")

                debug_exists = os.path.exists(debug_dir)
                release_exists = os.path.exists(release_dir)

                # Find *.so file
                so_file = self._find_so_file(cargo_target)
                has_so_file = so_file is not None

                if debug_exists:
                    # Use ├─ if there are more items below
                    symbol = "├─" if release_exists or has_so_file else "└─"
                    try:
                        debug_mtime = os.path.getmtime(debug_dir)
                        debug_time = self._format_timestamp_pdt(debug_mtime)
                        print(f"  {symbol} Debug:   {debug_dir} (modified: {debug_time})")
                    except OSError:
                        print(f"  {symbol} Debug:   {debug_dir} (unable to read timestamp)")

                if release_exists:
                    # Use ├─ if there's a *.so file below
                    symbol = "├─" if has_so_file else "└─"
                    try:
                        release_mtime = os.path.getmtime(release_dir)
                        release_time = self._format_timestamp_pdt(release_mtime)
                        print(f"  {symbol} Release: {release_dir} (modified: {release_time})")
                    except OSError:
                        print(f"  {symbol} Release: {release_dir} (unable to read timestamp)")

                # Show *.so file if found
                if has_so_file:
                    try:
                        so_mtime = os.path.getmtime(so_file)
                        so_time = self._format_timestamp_pdt(so_mtime)
                        print(f"  └─ Binary:  {so_file} (modified: {so_time})")
                    except OSError:
                        print(f"  └─ Binary:  {so_file} (unable to read timestamp)")

        return results

    def _show_relevant_pth_files(self):
        """Show .pth files that are relevant to dynamo imports."""
        # Get site-packages directories
        import site
        site_packages_dirs = site.getsitepackages()
        if hasattr(site, 'getusersitepackages'):
            site_packages_dirs.append(site.getusersitepackages())

        if not site_packages_dirs:
            return

        # Show site-packages location first
        main_site_packages = site_packages_dirs[0] if site_packages_dirs else None

        pth_files = []
        dist_info_dirs = []

        for site_dir in site_packages_dirs:
            if not os.path.exists(site_dir):
                continue

            try:
                # Find .pth files
                for file in os.listdir(site_dir):
                    if file.endswith('.pth') and ('dynamo' in file.lower() or 'ai_dynamo' in file.lower()):
                        pth_path = os.path.join(site_dir, file)
                        try:
                            mtime = os.path.getmtime(pth_path)
                            # Read the content to see what path it adds
                            with open(pth_path, 'r') as f:
                                content = f.read().strip()
                            pth_files.append((pth_path, mtime, content))
                        except OSError:
                            pass

                    # Find distribution info directories
                    if (file.endswith('.dist-info') and
                        ('dynamo' in file.lower() or 'ai_dynamo' in file.lower())):
                        dist_info_path = os.path.join(site_dir, file)
                        if os.path.isdir(dist_info_path):
                            dist_info_dirs.append(dist_info_path)

            except OSError:
                continue

        if pth_files or dist_info_dirs:
            # Determine if every path is under a known site-packages dir
            def _is_under_any_site_dir(path: str) -> bool:
                try:
                    for sdir in site_packages_dirs:
                        sdir_norm = os.path.join(sdir, '')
                        if path.startswith(sdir_norm):
                            return True
                except Exception:
                    pass
                return False

            all_under_site = True
            for d in dist_info_dirs:
                if not _is_under_any_site_dir(d):
                    all_under_site = False
                    break
            if all_under_site:
                for pth_path, _, _ in pth_files:
                    if not _is_under_any_site_dir(pth_path):
                        all_under_site = False
                        break

            # Print header without a leading blank line
            print("Python package installation status:")

            # Only print the absolute Site-packages root if entries won't already
            # include the normalized "site-packages/..." prefix
            if (main_site_packages and not all_under_site):
                print(f"Site-packages: {main_site_packages}")

            # Normalize and print distribution info similar to Runtime/Framework sections
            entries = []
            for dist_info_dir in dist_info_dirs:
                dist_name = os.path.basename(dist_info_dir)
                try:
                    # Determine install type (editable vs wheel)
                    direct_url_path = os.path.join(dist_info_dir, "direct_url.json")
                    install_type = None
                    if os.path.exists(direct_url_path):
                        try:
                            with open(direct_url_path, 'r') as f:
                                direct_url_data = json.load(f)
                                is_editable = (direct_url_data.get("editable") or
                                               (direct_url_data.get("dir_info", {}).get("editable")))
                                install_type = ".pth (editable)" if is_editable else ".whl (wheel)"
                        except (json.JSONDecodeError, OSError):
                            pass
                    # Fallback when direct_url.json missing
                    if install_type is None:
                        install_type = ".whl (wheel)"

                    # Created time
                    try:
                        ctime = os.path.getctime(dist_info_dir)
                        created_time = self._format_timestamp_pdt(ctime)
                    except OSError:
                        created_time = None

                    package_name = dist_name.replace('.dist-info', '').replace('_', '-')
                    entries.append({
                        'name': package_name,
                        'path': dist_info_dir,
                        'created': created_time,
                        'type': install_type,
                    })
                except OSError:
                    continue

            # Sort runtime first, then others alphabetically
            def sort_key(e):
                name = e['name']
                return (0 if name.startswith('ai-dynamo-runtime') else 1, name)

            entries.sort(key=sort_key)

            # Determine alignment width like component lists
            max_width = max((len(e['name']) for e in entries), default=0)

            for e in entries:
                created_suffix = f" (created: {e['created']})" if e['created'] else ""
                # Normalize paths under site-packages to a concise relative form
                display_path = e['path']
                try:
                    for site_dir in site_packages_dirs:
                        # Ensure trailing slash match semantics
                        site_dir_norm = os.path.join(site_dir, '')
                        path_norm = os.path.join(display_path, '')
                        if display_path.startswith(site_dir_norm):
                            rel = os.path.relpath(display_path, site_dir)
                            display_path = f"site-packages/{rel}"
                            break
                except Exception:
                    pass
                # Match indentation: checkmark, padded name, then normalized path; include created time at the end
                print(f"   ✅ {e['name']:<{max_width}} {display_path}{created_suffix}")

            # Show .pth files (only present in editable installs)
            for pth_path, mtime, content in pth_files:
                timestamp_str = self._format_timestamp_pdt(mtime)
                # Normalize pth path relative to a known site-packages dir
                display_pth = pth_path
                try:
                    for site_dir in site_packages_dirs:
                        site_dir_norm = os.path.join(site_dir, '')
                        if pth_path.startswith(site_dir_norm):
                            rel = os.path.relpath(pth_path, site_dir)
                            display_pth = f"site-packages/{rel}"
                            break
                except Exception:
                    pass
                print(f"      {display_pth} (modified: {timestamp_str})")
                print(f"      └─ Points to: {content}")

    # ====================================================================
    # USAGE EXAMPLES AND GUIDANCE
    # ====================================================================

    def show_usage_examples(self):
        """Show practical usage examples.

        Prints formatted examples of common dynamo operations including:
        - Starting frontend server
        - Starting vLLM backend
        - Making inference requests
        - Setting up development environment
        - Building packages

        Console output example:
            Usage Examples
            ========================================

            1. Start Frontend Server:
               python -m dynamo.frontend --http-port 8000

            2. Start vLLM Backend:
               python -m dynamo.vllm --model Qwen/Qwen2.5-0.5B
               ...
        """
        print("""
Usage Examples
========================================

1. Start Frontend Server:
   python -m dynamo.frontend --http-port 8000

2. Start vLLM Backend:
   python -m dynamo.vllm --model Qwen/Qwen2.5-0.5B

3. Send Inference Request:
   curl -X POST http://localhost:8000/v1/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model": "Qwen/Qwen2.5-0.5B", "prompt": "Hello", "max_tokens": 50}'

4. For local development: Set PYTHONPATH to use workspace sources without rebuilding:
   • Discover what PYTHONPATH to set: ./bin/dynamo_check.py --try-pythonpath --imports""")
        if self.workspace_dir:
            print(f"   • Then set in your shell: export PYTHONPATH=\"{self._get_pythonpath()}\"")
        else:
            print("   • Then set in your shell: export PYTHONPATH=\"$HOME/dynamo/components/*/src\"")

        not_found_suffix = "" if self._is_dynamo_build_available() else " (dynamo_build.sh not found)"
        print(f"""
5. Build Packages:
   dynamo_build.sh --dev              # Development mode{not_found_suffix}
   dynamo_build.sh --release          # Production wheels{not_found_suffix}""")

    def _get_pythonpath(self) -> str:
        """Generate PYTHONPATH recommendation string.

        Returns:
            Colon-separated string of component source paths
            Example: '~/dynamo/components/frontend/src:~/dynamo/components/planner/src:~/dynamo/components/backends/vllm/src'

        Note: Scans workspace for all component src directories and joins them for PYTHONPATH usage.
        """
        paths = []
        if not self.workspace_dir:
            return ""

        # Collect all component source paths
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                src_path = f"{comp_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)

        # Collect all backend source paths
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                src_path = f"{backend_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)

        return ":".join(paths)

    # ====================================================================
    # TROUBLESHOOTING AND SUMMARY
    # ====================================================================

    def show_troubleshooting(self):
        """Show troubleshooting guidance only if there were import failures."""
        # Check if any imports failed
        import_results = self.results.get('imports', {})
        failed_imports = [component for component, result in import_results.items()
                         if result.startswith('❌')]

        if not failed_imports:
            return  # No failures, skip troubleshooting section

        not_found_suffix = "" if self._is_dynamo_build_available() else "  # (dynamo_build.sh not found)"
        troubleshooting_msg = f"""
Troubleshooting
========================================

Found {len(failed_imports)} failed import(s). Common Issues:
1. ImportError for framework components:
   $ export PYTHONPATH=...

2. Package not found:
   $ dynamo_build.sh --release{not_found_suffix}

3. Check current status:
   $ dynamo_build.sh --check{not_found_suffix}"""

        print(troubleshooting_msg)

        if not self.workspace_dir:
            print("""
⚠️  Workspace not found!
   → Ensure you're running from a dynamo workspace
   → Expected locations: ~/dynamo, /workspace, ~/dynamo""")

    def show_summary(self):
        """Show comprehensive summary."""
        print("\nSummary")
        print("=" * 40)

        # Import status
        import_results = self.results.get('imports', {})
        if import_results:
            total = len(import_results)
            passed = sum(1 for r in import_results.values() if r.startswith('✅'))
            if passed == total:
                print(f"✅ Import tests: {passed}/{total} passed")
            else:
                print(f"❌ Import tests: {passed}/{total} passed")

    # ====================================================================
    # MAIN ORCHESTRATION
    # ====================================================================

    def run_all(self):
        """Run comprehensive check with all functionality.

        Performs complete dynamo package validation including:
        - Component discovery and import testing
        - Usage examples and troubleshooting guidance
        - Summary of results

        Console output example:
            Dynamo Comprehensive Check
            ============================================================
            Runtime components (ai-dynamo-runtime 0.4.0):
               ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               ✅ dynamo.llm          /opt/dynamo/venv/lib/.../dynamo/llm/__init__.py

            Framework components (ai-dynamo 0.4.0):
               ✅ dynamo.frontend     /opt/dynamo/venv/lib/.../dynamo/frontend/__init__.py

            Usage Examples
            ========================================
            1. Start Frontend Server:
               python -m dynamo.frontend --http-port 8000
               ...

            Summary
            ========================================
            ✅ Import tests: 5/5 passed
        """
        print("Dynamo Comprehensive Check")
        print("=" * 60)

        # Execute all checks (package versions now shown in import testing headers)
        self.results['imports'] = self.test_imports()

        # Check if there were any import failures
        import_results = self.results.get('imports', {})
        has_failures = any(result.startswith('❌') for result in import_results.values())

        # Provide guidance
        if not has_failures:
            self.show_usage_examples()
        self.show_troubleshooting()
        self.show_summary()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive dynamo package checker")
    parser.add_argument('--imports', action='store_true', help='Only test imports')
    parser.add_argument('--examples', action='store_true', help='Only show examples')
    parser.add_argument('--try-pythonpath', action='store_true', help='Test imports with workspace component source directories in sys.path')

    args = parser.parse_args()
    checker = DynamoChecker()

    # Set up sys.path if requested
    if args.try_pythonpath:
        checker._setup_pythonpath()

    if args.imports:
        checker.test_imports()
    elif args.examples:
        checker.show_usage_examples()
    else:
        checker.run_all()


if __name__ == "__main__":
    main()
