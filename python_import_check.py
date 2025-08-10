#!/usr/bin/env python3
"""
Comprehensive dynamo package checker, tester, and usage guide.

Combines version checking, import testing, and usage examples into a single tool.
Features dynamic component discovery and comprehensive troubleshooting guidance.

Usage:
    ./bin/python_import_check.py              # Run all checks
    ./bin/python_import_check.py --imports    # Only test imports
    ./bin/python_import_check.py --versions   # Only check versions
    ./bin/python_import_check.py --examples   # Only show examples
    ./bin/python_import_check.py --help       # Show help
"""

import sys
import os
import argparse
import asyncio
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Tuple, Any


class DynamoChecker:
    """Comprehensive dynamo package checker."""

    def __init__(self):
        self.workspace_dir = self._find_workspace()
        self.results = {}

    # ====================================================================
    # WORKSPACE AND COMPONENT DISCOVERY
    # ====================================================================

    def _find_workspace(self) -> str:
        """Find dynamo workspace directory."""
        candidates = [
            os.path.expanduser("~/dynamo"),
            "/workspace",
            "~/dynamo"
        ]

        for candidate in candidates:
            if os.path.exists(candidate) and os.path.exists(f"{candidate}/README.md"):
                return candidate
        return ""

    def _discover_runtime_components(self) -> List[str]:
        """Discover ai-dynamo-runtime components from filesystem."""
        components = ["dynamo._core"]  # Always include compiled Rust module

        if not self.workspace_dir:
            return components

        runtime_path = f"{self.workspace_dir}/lib/bindings/python/src/dynamo"
        if not os.path.exists(runtime_path):
            return components

        for item in os.listdir(runtime_path):
            item_path = os.path.join(runtime_path, item)
            if os.path.isdir(item_path) and os.path.exists(f"{item_path}/__init__.py"):
                components.append(f"dynamo.{item}")

        return components

    def _discover_framework_components(self) -> List[str]:
        """Discover ai-dynamo framework components from filesystem."""
        components = []

        if not self.workspace_dir:
            return components

        # Scan direct components (frontend, planner, etc.)
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                item_path = os.path.join(comp_path, item)
                if (os.path.isdir(item_path) and
                    os.path.exists(f"{item_path}/src/dynamo/{item}/__init__.py")):
                    components.append(f"dynamo.{item}")

        # Scan backend components (vllm, sglang, etc.)
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                item_path = os.path.join(backend_path, item)
                if (os.path.isdir(item_path) and
                    os.path.exists(f"{item_path}/src/dynamo/{item}/__init__.py")):
                    components.append(f"dynamo.{item}")

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

        # Collect backend source paths
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                src_path = f"{backend_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)

        # Update PYTHONPATH environment
        if paths:
            current_path = os.environ.get("PYTHONPATH", "")
            new_path = ":".join(paths)
            if current_path:
                new_path = f"{new_path}:{current_path}"
            os.environ["PYTHONPATH"] = new_path

    # ====================================================================
    # PACKAGE AND IMPORT TESTING
    # ====================================================================

    def check_package_versions(self) -> Dict[str, str]:
        """Check version information for main dynamo packages."""
        print("Package Versions")
        print("=" * 40)

        results = {}
        packages = [
            ('ai-dynamo', 'Complete distributed inference framework'),
            ('ai-dynamo-runtime', 'Core runtime with Rust extensions'),
        ]

        for pkg_name, description in packages:
            try:
                version = importlib.metadata.version(pkg_name)
                result = f"✅ {version}"
                print(f"   {pkg_name}: {version}")
            except importlib.metadata.PackageNotFoundError:
                result = "❌ Not installed"
                print(f"   {pkg_name}: Not installed")
            except Exception as e:
                result = f"⚠️ Error: {e}"
                print(f"   {pkg_name}: Error - {e}")

            results[pkg_name] = result

        return results

    def test_imports(self) -> Dict[str, str]:
        """Test imports for all discovered components."""
        print("\nImport Testing")
        print("=" * 40)

        results = {}

        # Test runtime components
        print("Runtime components (ai-dynamo-runtime):")
        runtime_components = self._discover_runtime_components()
        for component in runtime_components:
            try:
                module = __import__(component, fromlist=[''])
                results[component] = "✅ Success"
                print(f"   ✅ {component}")
            except ImportError as e:
                results[component] = f"❌ Failed: {e}"
                print(f"   ❌ {component}: {e}")

        # Test framework components
        print("\nFramework components (ai-dynamo):")
        framework_components = self._discover_framework_components()
        for component in framework_components:
            try:
                module = __import__(component, fromlist=[''])
                results[component] = "✅ Success"
                print(f"   ✅ {component}")
            except ImportError as e:
                results[component] = f"❌ Failed: {e}"
                print(f"   ❌ {component}: {e}")

        return results

    # ====================================================================
    # USAGE EXAMPLES AND GUIDANCE
    # ====================================================================

    def show_usage_examples(self):
        """Show practical usage examples."""
        print("\nUsage Examples")
        print("=" * 40)

        print("\n1. Start Frontend Server:")
        print("   python -m dynamo.frontend --http-port 8000")

        print("\n2. Start vLLM Backend:")
        print("   python -m dynamo.vllm --model Qwen/Qwen2.5-0.5B")

        print("\n3. Send Inference Request:")
        print("   curl -X POST http://localhost:8000/v1/completions \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"model\": \"Qwen/Qwen2.5-0.5B\", \"prompt\": \"Hello\", \"max_tokens\": 50}'")

        print("\n4. Optional Development Setup (hot-reload all the ai-dynamo components, no build required):")
        if self.workspace_dir:
            print(f"   export PYTHONPATH=\"{self._get_pythonpath()}\"")
        else:
            print("   export PYTHONPATH=\"$HOME/dynamo/components/*/src\"")

        print("\n5. Build Packages:")
        print("   pybuild.sh --dev              # Development mode")
        print("   pybuild.sh --release          # Production wheels")

    def _get_pythonpath(self) -> str:
        """Generate PYTHONPATH recommendation string."""
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

        print("\nTroubleshooting")
        print("=" * 40)

        print(f"\nFound {len(failed_imports)} failed import(s). Common Issues:")
        print("1. ImportError for runtime components:")
        print("   → Run: pybuild.sh --dev")

        print("\n2. ImportError for framework components:")
        print("   → Set PYTHONPATH to include component src directories")

        print("\n3. Package not found:")
        print("   → Run: pybuild.sh --release")

        print("\n4. Check current status:")
        print("   → Run: pybuild.sh --check")

        if not self.workspace_dir:
            print("\n⚠️  Workspace not found!")
            print("   → Ensure you're running from a dynamo workspace")
            print("   → Expected locations: ~/dynamo, /workspace, ~/dynamo")

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
        """Run comprehensive check with all functionality."""
        print("Dynamo Comprehensive Check")
        print("=" * 60)

        self._setup_pythonpath()

        # Execute all checks
        self.results['packages'] = self.check_package_versions()
        self.results['imports'] = self.test_imports()

        # Provide guidance
        self.show_usage_examples()
        self.show_troubleshooting()
        self.show_summary()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive dynamo package checker")
    parser.add_argument('--imports', action='store_true', help='Only test imports')
    parser.add_argument('--versions', action='store_true', help='Only check versions')
    parser.add_argument('--examples', action='store_true', help='Only show examples')

    args = parser.parse_args()
    checker = DynamoChecker()

    # Set up environment
    checker._setup_pythonpath()

    if args.imports:
        checker.test_imports()
    elif args.versions:
        checker.check_package_versions()
    elif args.examples:
        checker.show_usage_examples()
    else:
        checker.run_all()


if __name__ == "__main__":
    main()
