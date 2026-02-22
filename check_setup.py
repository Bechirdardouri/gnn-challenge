#!/usr/bin/env python3
"""Quick setup verification script for HeteroShot challenge.

This script checks that your environment is correctly set up.
"""

import sys
import subprocess


def check_python_version():
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f" Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name):
    """Check if a package is importable."""
    try:
        __import__(package_name)
        print(f" {package_name}")
        return True
    except ImportError:
        print(f" {package_name} not installed")
        return False


def check_files():
    """Check required files exist."""
    from pathlib import Path
    
    required = [
        "data/train.csv",
        "data/val.csv", 
        "data/test.csv",
        "data/edges.csv"
    ]
    
    all_exist = True
    for file in required:
        if Path(file).exists():
            print(f" {file}")
        else:
            print(f" {file} missing - run: python data/make_dataset.py")
            all_exist = False
    
    return all_exist


def main():
    """Run all checks."""
    print(" HeteroShot Environment Check\n")
    
    print(" Python Version:")
    py_ok = check_python_version()
    print()
    
    print(" Required Packages:")
    packages = ["numpy", "pandas", "sklearn", "torch", "torch_geometric"]
    pkg_ok = all(check_package(pkg) for pkg in packages)
    print()
    
    print(" Dataset Files:")
    files_ok = check_files()
    print()
    
    if py_ok and pkg_ok and files_ok:
        print(" All checks passed! You're ready to compete!")
        print("\nNext steps:")
        print("  1. Run baseline: python starter_code/baseline_tabular.py")
        print("  2. Create your model")
        print("  3. Submit via Pull Request")
        return 0
    else:
        print("  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
