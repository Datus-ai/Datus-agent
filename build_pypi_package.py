#!/usr/bin/env python3
"""
Build and publish script for datus-agent package
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: str, check: bool = True, capture_output: bool = False, description: str = "", verbose: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    if description:
        print(f"🔄 {description}")
    else:
        print(f"🔄 Running: {cmd}")

    try:
        # Add --verbose flag only if supported and requested
        if verbose and "--verbose" not in cmd:
            full_cmd = f"{cmd} --verbose"
        else:
            full_cmd = cmd

        result = subprocess.run(full_cmd, shell=True, capture_output=capture_output, text=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}")
            if capture_output:
                if result.stdout:
                    print(f"📤 stdout: {result.stdout}")
                if result.stderr:
                    print(f"📥 stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        else:
            print("✅ Command completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command execution failed: {e}")
        if check:
            sys.exit(1)
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if check:
            sys.exit(1)
        raise


def clean_build():
    """Clean build artifacts"""
    print("🧹 Cleaning build artifacts...")
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    cleaned_count = 0

    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  📁 Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"  📄 Removed file: {path}")
                cleaned_count += 1
            except Exception as e:
                print(f"  ⚠️  Failed to remove {path}: {e}")

    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} build artifacts")
    else:
        print("ℹ️  No build artifacts found to clean")


def build_package():
    """Build the package"""
    print("🔨 Building package...")
    run_command("python3 -m build", description="Building Python package")


def install_locally():
    """Install the package locally"""
    print("📦 Installing package locally...")
    run_command("pip3 install -e .", description="Installing package in development mode")


def install_from_dist():
    """Install from built distribution"""
    print("📦 Installing from built distribution...")
    # Find the latest wheel file
    dist_dir = Path("dist")
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel files found in dist/")
        print("💡 Try running 'build' command first")
        return

    latest_wheel = max(wheel_files, key=lambda x: x.stat().st_mtime)
    print(f"📦 Installing wheel: {latest_wheel.name}")
    run_command(f"pip3 install {latest_wheel}", description=f"Installing {latest_wheel.name}")


def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    try:
        result = run_command(
            "python3 -c 'import datus; print(f\"Datus version: {datus.__version__}\")'",
            capture_output=True,
            description="Testing package import and version",
        )
        if result.stdout:
            print(f"📤 {result.stdout.strip()}")
        print("✅ Installation test passed!")
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def upload_to_test_pypi():
    """Upload to Test PyPI"""
    print("🚀 Uploading to Test PyPI...")
    run_command("python3 -m twine upload --repository testpypi dist/*", description="Uploading to Test PyPI")


def upload_to_pypi():
    """Upload to PyPI"""
    print("🚀 Uploading to PyPI...")
    run_command("python3 -m twine upload dist/*", description="Uploading to PyPI")


def check_package():
    """Check the package before upload"""
    print("🔍 Checking package...")
    run_command("python3 -m twine check dist/*", description="Validating package distribution files", verbose=False)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🚀 Datus Agent Package Builder")
        print("=" * 40)
        print("Usage: python build_package.py <command>")
        print("\n📋 Available Commands:")
        print("  🧹 clean       - Clean build artifacts")
        print("  🔨 build       - Build the package")
        print("  📦 install     - Install locally (development mode)")
        print("  📦 install-dist - Install from built distribution")
        print("  🧪 test        - Test installation")
        print("  🔍 check       - Check package before upload")
        print("  🚀 upload-test - Upload to Test PyPI")
        print("  🚀 upload      - Upload to PyPI")
        print("  🔄 all         - Clean, build, check, and test")
        print("  📤 publish     - Clean, build, check, and upload to PyPI")
        print("\n💡 Examples:")
        print("  python build_package.py build")
        print("  python build_package.py all")
        print("  python build_package.py publish")
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "clean":
            clean_build()
        elif command == "build":
            build_package()
        elif command == "install":
            install_locally()
        elif command == "install-dist":
            install_from_dist()
        elif command == "test":
            test_installation()
        elif command == "check":
            check_package()
        elif command == "upload-test":
            upload_to_test_pypi()
        elif command == "upload":
            upload_to_pypi()
        elif command == "all":
            print("🔄 Running full build and test workflow...")
            clean_build()
            build_package()
            check_package()
            if test_installation():
                print("🎉 All tasks completed successfully!")
            else:
                print("⚠️  Build completed but installation test failed")
        elif command == "publish":
            print("🚀 Running full publish workflow...")
            clean_build()
            build_package()
            check_package()
            upload_to_pypi()
            print("🎉 Package published successfully!")
        else:
            print(f"❌ Unknown command: {command}")
            print("💡 Run without arguments to see available commands")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
