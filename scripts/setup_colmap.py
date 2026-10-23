"""
COLMAP installation helper for Windows.

Checks if COLMAP is installed and provides installation guidance.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def check_colmap() -> bool:
    """Check if COLMAP is available on PATH."""
    colmap = shutil.which("colmap")
    if colmap:
        print(f"✅ COLMAP found: {colmap}")
        try:
            result = subprocess.run(["colmap", "help"], capture_output=True, text=True, timeout=10)
            print(f"   Version info available.")
        except Exception:
            pass
        return True

    print("❌ COLMAP not found on PATH.")
    return False


def print_install_instructions():
    """Print platform-specific COLMAP installation instructions."""
    os_name = platform.system()

    print("\n" + "=" * 60)
    print("COLMAP Installation Guide")
    print("=" * 60)

    if os_name == "Windows":
        print("""
Option 1: Pre-built binary (Recommended)
  1. Download from: https://github.com/colmap/colmap/releases
  2. Extract to C:\\COLMAP\\
  3. Add C:\\COLMAP\\ to your PATH environment variable

Option 2: Via conda
  conda install -c conda-forge colmap

Option 3: Build from source
  See: https://colmap.github.io/install.html
""")
    elif os_name == "Linux":
        print("""
Option 1: Package manager
  sudo apt-get install colmap

Option 2: Via conda
  conda install -c conda-forge colmap

Option 3: Build from source
  See: https://colmap.github.io/install.html
""")
    elif os_name == "Darwin":
        print("""
Option 1: Homebrew
  brew install colmap

Option 2: Via conda
  conda install -c conda-forge colmap
""")

    print("After installation, verify with: colmap help")
    print("=" * 60)


if __name__ == "__main__":
    if not check_colmap():
        print_install_instructions()
        sys.exit(1)
