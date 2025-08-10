#!/usr/bin/env python3
"""
Comprehensive dynamo package checker, tester, and usage guide.

Combines version checking, import testing, and usage examples into a single tool.
Features dynamic component discovery and comprehensive troubleshooting guidance.

Usage:
    ./bin/python_import_check.py                        # Run all checks
    ./bin/python_import_check.py --imports              # Only test imports
    ./bin/python_import_check.py --examples             # Only show examples
    ./bin/python_import_check.py --try-pythonpath      # Test imports with workspace paths
    ./bin/python_import_check.py --help                 # Show help
"""

import sys
import os
import argparse
import asyncio
import importlib.metadata
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any


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
                    if self.workspace_dir and module_path.startswith(self.workspace_dir):
                        # From workspace source
                        rel_path = os.path.relpath(module_path, self.workspace_dir)
                        print(f"   ✅ {component:<{max_width}} {rel_path}")
                    elif site_packages and module_path.startswith(site_packages):
                        # From installed package - show full path
                        print(f"   ✅ {component:<{max_width}} {module_path}")
                    else:
                        # Other location
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

        # Show PYTHONPATH recommendation if any framework components failed
        if framework_failures and self.workspace_dir:
            pythonpath = self._get_pythonpath()
            if pythonpath:
                print("\nMissing framework components detected. To resolve this, choose one of the following options:")
                print("1. For local development, set the PYTHONPATH environment variable:")
                print(f"   ./bin/python_import_check.py --try-pythonpath --imports\n   export PYTHONPATH=\"{pythonpath}\"")
                print("2. For a production-release (slower build time), build the packages with:")
                print("   pybuild.sh --release")

        return results

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
   • Discover what PYTHONPATH to set: ./bin/python_import_check.py --try-pythonpath --imports""")
        if self.workspace_dir:
            print(f"   • Then set in your shell: export PYTHONPATH=\"{self._get_pythonpath()}\"")
        else:
            print("   • Then set in your shell: export PYTHONPATH=\"$HOME/dynamo/components/*/src\"")

        print("""
5. Build Packages:
   pybuild.sh --dev              # Development mode
   pybuild.sh --release          # Production wheels""")

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

        print(f"""
Troubleshooting
========================================

Found {len(failed_imports)} failed import(s). Common Issues:
1. ImportError for framework components:
   $ export PYTHONPATH=...

2. Package not found:
   $ pybuild.sh --release

3. Check current status:
   $ pybuild.sh --check""")

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
