"""
pytest configuration for stage tests

This sets up the Python path and environment for testing.
"""
import sys
import os

# Get the binary directory (where DLLs and Python modules are)
binary_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Binaries', 'Debug'))
rznode_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Core', 'rznode', 'python'))

# Add to path
sys.path.insert(0, binary_dir)
sys.path.insert(0, rznode_python)

# Set environment for USD DLLs
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = binary_dir

# Change working directory to binary_dir so DLLs can be loaded
os.chdir(binary_dir)

print(f"Test environment configured:")
print(f"  Binary dir: {binary_dir}")
print(f"  RZNode Python: {rznode_python}")
print(f"  Working dir: {os.getcwd()}")
